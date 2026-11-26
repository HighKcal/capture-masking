"""Exercise geometry using the original product trace and a fresh detail trace."""

import copy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image

from detail_geometry import FRAME_MISMATCH, GEOMETRY_SOURCE, UnverifiedGeometry, VERSION, digest, repair


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'email-tail-fix-v0'


class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        records = (
            ('user', 'user-chat-01-rgb.png', 'evidence/analysis/after-01/document.json', 26),
            ('synthetic', 'synthetic-email.png', 'evidence/native/native-01/analysis-03-synthetic-email.json', 3),
        )
        for name, image, doc, index in records:
            blobs = json.loads((PRIOR / doc).read_bytes())['blobs']
            fresh = json.loads((HERE / 'evidence/detail-before-01' / f'{name}.json').read_bytes())['detail']
            with Image.open(PRIOR / 'inputs' / image) as original:
                cls.cases[name] = original.copy(), blobs['ocr'], blobs['detail'], fresh, index

    def test_old_and_fresh_reproduction(self):
        for name, (rgb, ocr, old, fresh, index) in self.cases.items():
            for raw in (old, fresh):
                with self.subTest(name=name, old=raw is old):
                    original_hash = digest(raw)
                    effective, receipt = repair(rgb, ocr, raw['identity'], raw)
                    self.assertEqual(original_hash, digest(raw))
                    self.assertEqual(receipt, repair(rgb, ocr, raw['identity'], raw)[1])
                    self.assertEqual(receipt['raw_sha256'], original_hash)
                    self.assertEqual(receipt['effective_sha256'], digest(effective))
                    self.assertEqual(receipt['version'], VERSION)
                    self.assertEqual(raw['status'], 'partial_failure')
                    self.assertEqual(effective['status'], 'ok')
                    self.assertEqual(receipt['repaired_count'], 1)
                    self.assertEqual(len(receipt['repaired_units']), 1)
                    self.assertEqual(receipt['repaired_units'][0]['order_index'], index)
                    self.assertEqual(receipt['repaired_units'][0]['reason'], FRAME_MISMATCH)
                    self.assertEqual(raw['units'][:index] + raw['units'][index + 1:],
                                     effective['units'][:index] + effective['units'][index + 1:])
                    self.assertEqual(raw['evidence'], effective['evidence'])
                    row = effective['units'][index]
                    self.assertEqual(''.join(token['text'] for token in row['tokens']), row['text'])
                    self.assertEqual({token['geometry_source'] for token in row['tokens']}, {GEOMETRY_SOURCE})
                    self.assertNotEqual(row['transform']['forward_crop_dimensions'],
                                        row['transform']['legacy_inverse_frame_dimensions'])
                    self.assert_crop_local_projection(row)

    def assert_crop_local_projection(self, row):
        transform = row['transform']
        width, height = transform['forward_crop_dimensions']
        quad = np.array(transform['forward_quad'], dtype=np.float32)
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        forward = cv2.getPerspectiveTransform(quad, rect)
        cells = np.array([token['polygon'] for token in row['tokens']], dtype=np.float64)
        mapped = cv2.perspectiveTransform(cells.reshape(1, -1, 2), forward).reshape(-1, 4, 2)
        np.testing.assert_allclose(mapped[:, :2, 1], 0, atol=1e-6)
        np.testing.assert_allclose(mapped[:, 2:, 1], height, atol=1e-6)
        np.testing.assert_allclose(mapped[:, 0, 0], mapped[:, 3, 0], atol=1e-6)
        np.testing.assert_allclose(mapped[:, 1, 0], mapped[:, 2, 0], atol=1e-6)
        self.assertTrue(np.all(mapped[:, 0, 0] < mapped[:, 1, 0]))

    def mutated_synthetic(self, mutate):
        rgb, ocr, old, _, index = self.cases['synthetic']
        raw = copy.deepcopy(old)
        mutate(raw, index)
        effective, proof = repair(rgb, ocr, raw['identity'], raw)
        self.assertEqual(proof['repaired_count'], 0)
        self.assertEqual(effective, raw)
        self.assertEqual(proof['raw_sha256'], proof['effective_sha256'])
        return proof['retained_uncertain']

    def test_crop_size_or_rotation_disagreement_stays_uncertain(self):
        def wrong_shape(raw, index):
            raw['units'][index]['transform']['mapped_crop_shape'][0] -= 1

        def rotation(raw, index):
            raw['units'][index]['transform']['forward_rot90'] = True

        def clipped(raw, index):
            raw['units'][index]['transform']['parent_was_clipped'] = True

        for change in (wrong_shape, rotation, clipped):
            with self.subTest(change=change.__name__):
                self.assertEqual(self.mutated_synthetic(change)[0]['reason'], 'actual_crop_frame_not_proven')

    def test_ctc_column_mutation_stays_uncertain(self):
        def change(raw, index):
            raw['evidence']['trace']['recognition']['ctc'][index]['columns'][-1] -= 1

        self.assertEqual(self.mutated_synthetic(change)[0]['reason'], 'ctc_word_info_correspondence')

    def test_ctc_confidence_mutation_stays_uncertain(self):
        def change(raw, index):
            raw['evidence']['trace']['recognition']['ctc'][index]['confidences'][-1] = 0

        self.assertEqual(self.mutated_synthetic(change)[0]['reason'], 'ctc_transcript_columns_or_scores')

    def test_inconsistent_route_stays_uncertain(self):
        def change(raw, index):
            raw['evidence']['trace']['before_score_filter']['boxes'][index][0][0] += 1

        self.assertEqual(self.mutated_synthetic(change)[0]['reason'], 'unverified_native_trace')

    def test_consistent_forged_polygon_still_fails_native_replay(self):
        def change(raw, index):
            trace = raw['evidence']['trace']
            for series in (trace['calculation']['raw_rows'], trace['calculation']['returned_rows'],
                           raw['transforms']['calculation']['raw_rows'], raw['transforms']['calculation']['returned_rows'],
                           trace['before_score_filter']['word_results'], raw['evidence']['public']['word_results']):
                series[index][0][2][0][0] += 1

        self.assertEqual(self.mutated_synthetic(change)[0]['reason'], 'native_polygon_replay_mismatch')

    def test_unknown_failure_stays_untouched(self):
        def change(raw, index):
            raw['units'][index]['transform']['reasons'].append('different_failure')

        self.assertEqual(self.mutated_synthetic(change), [])

    def test_noop_keeps_all_existing_good_units(self):
        rgb, ocr, old, _, _ = self.cases['synthetic']
        fixed, _ = repair(rgb, ocr, old['identity'], old)
        unchanged, proof = repair(rgb, ocr, old['identity'], fixed)
        self.assertEqual(unchanged, fixed)
        self.assertIsNot(unchanged, fixed)
        self.assertEqual(proof['repaired_units'], [])

    def test_changed_rgb_or_ocr_binding_rejected(self):
        rgb, ocr, old, _, _ = self.cases['synthetic']
        wrong_pixels = rgb.copy()
        wrong_pixels.putpixel((0, 0), (0, 0, 0))
        with self.assertRaisesRegex(UnverifiedGeometry, 'detail_geometry_rgb'):
            repair(wrong_pixels, ocr, old['identity'], old)
        wrong_ocr = dict(ocr, run_id='unrelated')
        with self.assertRaisesRegex(UnverifiedGeometry, 'detail_geometry_identity'):
            repair(rgb, wrong_ocr, old['identity'], old)


if __name__ == '__main__':
    unittest.main()
