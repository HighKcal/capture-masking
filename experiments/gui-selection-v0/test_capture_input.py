"""Model-free tests for capture ROI ownership and shared GUI snapshots."""

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
from PIL import Image

from capture_input import CapturedInput, CaptureInputError, thaw_metadata
from fresh_controller import FreshController
from fresh_protocol import EVENT_SCHEMA, MAX_SIDE, atomic_json, canonical, read_json, validate_request
import fresh_storage
from fresh_storage import Storage, StorageError


ATTEMPT_ID = 'a' * 32
SCREEN_GENERATION = 'b' * 64


def sample_rgb():
    return np.arange(45, dtype=np.uint8).reshape(3, 5, 3)


def payload(rgb=None, *, metadata=None, attempt=ATTEMPT_ID):
    return CapturedInput.from_rgb(sample_rgb() if rgb is None else rgb,
                                  attempt, SCREEN_GENERATION, metadata)


class CapturedInputTests(unittest.TestCase):
    def test_strided_crop_and_metadata_are_detached_and_immutable(self):
        frame = np.empty((10, 12, 3), dtype=np.uint8)
        roi = frame[2:5, 8:3:-1]
        roi[:] = sample_rgb()
        frame_ref = weakref.ref(frame)
        source = {'rect': [4, 2, 9, 5], 'screen': {'size': (12, 10)}}
        captured = payload(roi, metadata=source)
        self.assertEqual((captured.width, captured.height), (5, 3))
        frame[:] = 255
        del roi, frame
        self.assertIsNone(frame_ref())
        self.assertEqual(captured.rgb_bytes, sample_rgb().tobytes())
        source['rect'][0] = 99
        source['screen'].clear()
        self.assertEqual(thaw_metadata(captured.metadata),
                         {'rect': [4, 2, 9, 5], 'screen': {'size': [12, 10]}})
        with self.assertRaises(FrozenInstanceError):
            captured.width = 99
        with self.assertRaises(TypeError):
            captured.metadata['screen']['size'] = (1, 1)

    def test_rejects_malformed_pixels_and_dimensions_before_copy(self):
        malformed = [np.empty((2, 2), dtype=np.uint8), np.empty((2, 2, 4), dtype=np.uint8),
                     np.empty((2, 2, 3), dtype=np.uint16)]
        for rgb in malformed:
            with self.subTest(shape=rgb.shape, dtype=rgb.dtype), \
                    self.assertRaisesRegex(CaptureInputError, '^capture_input_invalid$'):
                payload(rgb)
        for shape in [(0, 5, 3), (1, MAX_SIDE + 1, 3), (MAX_SIDE, 1025, 3)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(CaptureInputError, '^image_limit$'):
                payload(np.broadcast_to(np.zeros((1, 1, 3), np.uint8), shape))
        for data in [b'\0' * 44, bytearray(45), memoryview(bytes(45))]:
            with self.subTest(data_type=type(data)), \
                    self.assertRaisesRegex(CaptureInputError, '^capture_input_invalid$'):
                CapturedInput(5, 3, data, ATTEMPT_ID, SCREEN_GENERATION)

    def test_rejects_bad_identity_and_unbounded_or_non_json_metadata(self):
        for attempt, generation in [('A' * 32, SCREEN_GENERATION), (ATTEMPT_ID, 'f' * 32)]:
            with self.assertRaisesRegex(CaptureInputError, '^capture_input_invalid$'):
                CapturedInput.from_rgb(sample_rgb(), attempt, generation)
        cycle = []
        cycle.append(cycle)
        invalid = [{'attempt_id': 'duplicate'}, {7: 'non-string key'}, {'bytes': bytes(45)},
                   {'array': sample_rgb()}, {'float': float('nan')}, {'integer': 2**64},
                   {'long': '가' * 1400}, {'many': [0] * 256}, {'cycle': cycle}]
        for index, metadata in enumerate(invalid):
            with self.subTest(index=index), self.assertRaisesRegex(CaptureInputError, '^capture_input_invalid$'):
                payload(metadata=metadata)


@unittest.skipUnless(sys.platform == 'win32', 'Windows file sharing required')
class CaptureSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='gui03 storage ')
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.root = self.directory / 'sessions'

    def store(self):
        storage = Storage(self.root)
        self.addCleanup(storage.close)
        return storage

    def test_file_and_crop_use_equal_rgb_png_and_distinct_provenance(self):
        storage = self.store()
        source = self.directory / 'synthetic.png'
        with Image.fromarray(sample_rgb()) as image:
            image.save(source, format='PNG')
        file_snapshot = storage.capture(source)
        with patch.object(fresh_storage, '_read_image') as read_file:
            crop_snapshot = storage.capture_rgb(payload(metadata={'rect': [0, 0, 5, 3]}))
        read_file.assert_not_called()
        self.assertNotEqual(crop_snapshot.image_id, file_snapshot.image_id)
        self.assertEqual(crop_snapshot.rgb.tobytes(), file_snapshot.rgb.tobytes())
        self.assertEqual(crop_snapshot.rgb_sha256, hashlib.sha256(sample_rgb().tobytes()).hexdigest())
        self.assertEqual(crop_snapshot.rgb_sha256, file_snapshot.rgb_sha256)
        self.assertEqual(crop_snapshot.file_sha256, file_snapshot.file_sha256)
        self.assertEqual(crop_snapshot.path.read_bytes(), source.read_bytes())
        self.assertEqual(file_snapshot.input_info(), dict(image_id=file_snapshot.image_id, kind='file', metadata={}))
        self.assertEqual(crop_snapshot.input_info(), dict(image_id=crop_snapshot.image_id, kind='capture',
            metadata=dict(rect=[0, 0, 5, 3], attempt_id=ATTEMPT_ID, screen_generation=SCREEN_GENERATION)))
        self.assertFalse(crop_snapshot.rgb.flags.writeable)
        with self.assertRaises(OSError):
            crop_snapshot.path.write_bytes(b'synthetic mutation')
        with self.assertRaises(OSError):
            crop_snapshot.path.unlink()
        with patch.object(fresh_storage, '_ordinary', side_effect=ValueError), \
                patch.object(storage, 'capture_rgb') as from_rgb, \
                self.assertRaisesRegex(StorageError, '^image_rejected$'):
            storage.capture(source)
        from_rgb.assert_not_called()

    def test_malformed_rgb_and_png_preparation_fail_without_snapshot_residue(self):
        storage = self.store()
        captured = payload()
        object.__setattr__(captured, 'rgb_bytes', b'\0')
        for invalid in [self.directory / 'image.png', captured]:
            with self.assertRaisesRegex(StorageError, '^capture_input_invalid$'):
                storage.capture_rgb(invalid)
        self.assertEqual(list(storage.session_dir.glob('image-*')), [])

        for operation, code in [
            (patch.object(Image.Image, 'save', side_effect=OSError), 'snapshot_failed'),
            (patch.object(Image, 'open', side_effect=SyntaxError), 'snapshot_changed'),
            (patch.object(fresh_storage, 'MAX_PNG_BYTES', 1), 'image_limit'),
        ]:
            with operation, self.assertRaisesRegex(StorageError, f'^{code}$'):
                storage.capture_rgb(payload())
            self.assertEqual(list(storage.session_dir.glob('image-*')), [])

        save = Image.Image.save

        def encode_wrong_pixels(_image, target, **options):
            with Image.new('RGB', (5, 3), 'red') as wrong:
                save(wrong, target, **options)

        with patch.object(Image.Image, 'save', new=encode_wrong_pixels), \
                self.assertRaisesRegex(StorageError, '^snapshot_changed$'):
            storage.capture_rgb(payload())
        self.assertEqual(list(storage.session_dir.glob('image-*')), [])
        self.assertTrue(storage.capture_rgb(payload()).path.is_file())

    def test_controller_retries_same_locked_crop_without_recapture(self):
        requests, ready, snapshots = [], [], []
        idle = {1: threading.Event(), 2: threading.Event()}
        source = payload(metadata={'screen': {'rect': [0, 0, 5, 3]}})
        source_ref = weakref.ref(source)

        def notify(kind, generation, value):
            if kind == 'fresh_ready':
                ready.append(value)
            if kind == 'fresh_idle':
                idle[generation].set()

        def launch(_argv, directory, _env):
            request = validate_request(read_json(directory / 'request.json'))
            requests.append(request)
            event = {key: request[key] for key in ('request_id', 'image_id', 'generation')}
            atomic_json(directory / 'event.json',
                        dict(event, schema=EVENT_SCHEMA, seq=1, stage='complete',
                             status='complete', result_sha256='c' * 64, result_bytes=1))
            process = Mock(finished=True)
            process.poll.return_value = 0
            return process

        def adopt(_directory, _request, snapshot, _event):
            snapshots.append(weakref.ref(snapshot))
            return object(), 0.0

        controller = FreshController(notify, storage_factory=lambda: Storage(self.root),
                                     process_factory=launch, consumer=adopt)
        try:
            controller.submit_capture(source, 1)
            del source
            self.assertTrue(idle[1].wait(5))
            self.assertIsNone(source_ref())
            self.assertTrue(controller.has_snapshot)
            self.assertGreater(ready[0]['timing']['input_prepare_seconds'], 0)
            first_info = ready[0]['input_info']
            self.assertEqual(first_info['kind'], 'capture')
            self.assertEqual(first_info['image_id'], requests[0]['image_id'])
            canonical(first_info)
            first_info['metadata']['screen']['rect'][0] = 99
            with patch.object(Storage, 'capture_rgb', side_effect=AssertionError('recapture')):
                self.assertTrue(controller.retry(2))
                self.assertTrue(idle[2].wait(5))
            self.assertEqual(len(ready), 2)
            self.assertEqual(ready[1]['timing']['input_prepare_seconds'], 0)
            self.assertEqual(ready[1]['input_info']['metadata']['screen']['rect'], [0, 0, 5, 3])
            self.assertEqual(requests[0]['image_id'], requests[1]['image_id'])
            self.assertEqual(requests[0]['snapshot'], requests[1]['snapshot'])
            self.assertNotEqual(requests[0]['request_id'], requests[1]['request_id'])
            self.assertIs(snapshots[0](), snapshots[1]())
            self.assertEqual(list(self.root.glob('session-*/image-*/request-*')), [])
        finally:
            controller.shutdown()
            self.assertTrue(controller.finished.wait(5))
        self.assertIsNone(controller.cleanup_error)
        self.assertIsNone(snapshots[0]())
        self.assertFalse(controller.has_snapshot)
        self.assertEqual(list(self.root.iterdir()), [])


class CaptureQueueTests(unittest.TestCase):
    def test_pending_crop_released_on_replacement_cancel_retire_and_close(self):
        for action in ('file', 'capture', 'cancel', 'retire', 'shutdown'):
            with self.subTest(action=action):
                self.exercise(action)

    def exercise(self, action):
        started, release, drained = threading.Event(), threading.Event(), threading.Event()
        acquired, discarded = [], []

        def capture_file(path):
            acquired.append(path.name)
            if path.name == 'first.png':
                started.set()
                release.wait(5)
            return path

        def capture_rgb(source):
            acquired.append(source.attempt_id)
            return source.attempt_id

        def notify(kind, generation, _value):
            if kind == 'fresh_idle' and generation != 1:
                drained.set()

        storage = Mock(capture=capture_file, capture_rgb=capture_rgb, discard=discarded.append)
        with patch.object(FreshController, '_run'):
            controller = FreshController(notify, storage_factory=lambda: storage)
            try:
                controller.submit('first.png', 1)
                self.assertTrue(started.wait(5))
                previous = payload()
                previous_ref = weakref.ref(previous)
                controller.submit_capture(previous, 2)
                del previous
                self.assertIsNotNone(previous_ref())
                if action == 'file':
                    controller.submit('last.png', 3)
                elif action == 'capture':
                    controller.submit_capture(payload(attempt='d' * 32), 3)
                else:
                    getattr(controller, action)()
                self.assertIsNone(previous_ref())
                self.assertEqual(acquired, ['first.png'])
                release.set()
                if action != 'shutdown':
                    self.assertTrue(drained.wait(5))
                    self.assertEqual(discarded, [Path('first.png')])
                last = {'file': ['last.png'], 'capture': ['d' * 32]}.get(action, [])
                self.assertEqual(acquired, ['first.png'] + last)
            finally:
                release.set()
                controller.shutdown()
                self.assertTrue(controller.finished.wait(5))
        storage.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
