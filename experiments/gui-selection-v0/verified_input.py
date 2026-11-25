"""Immutable RGB and provenance for an eligible input independent of analysis."""

from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType

import numpy as np

from capture_input import thaw_metadata
from fresh_protocol import MAX_PIXELS, MAX_SIDE, ProtocolError, canonical, hex_value


@dataclass(frozen=True, eq=False, slots=True)
class ValidatedInput:
    image_id: str
    file_sha256: str
    rgb_sha256: str
    width: int
    height: int
    kind: str
    _pixels: bytes = field(repr=False)
    _metadata: bytes = field(repr=False)

    @classmethod
    def from_snapshot(cls, snapshot):
        proof = cls(snapshot.image_id, snapshot.file_sha256, snapshot.rgb_sha256,
                    snapshot.width, snapshot.height, snapshot.kind, snapshot.rgb.tobytes(),
                    canonical(thaw_metadata(snapshot.metadata)))
        proof.validate()
        return proof

    @classmethod
    def from_document(cls, document):
        document.validate()
        identity = document.analysis['identity']
        if identity['image_id'] != document.key:
            raise ProtocolError('input_proof_invalid')
        proof = cls(document.key, identity['file_sha256'], identity['rgb_sha256'],
                    identity['width'], identity['height'], 'saved', document.rgb.tobytes(), b'{}')
        proof.validate()
        return proof

    @property
    def metadata(self):
        # This copy can be inspected without modifying the retained provenance.
        return MappingProxyType(self.input_info()['metadata'])

    def input_info(self):
        from fresh_protocol import parse_json

        return dict(image_id=self.image_id, kind=self.kind, metadata=parse_json(self._metadata))

    def validate(self):
        if (type(self.image_id) is not str or not self.image_id
                or self.kind not in ('file', 'capture', 'saved')
                or (self.kind != 'saved' and not hex_value(self.image_id, 32))
                or not hex_value(self.file_sha256, 64) or not hex_value(self.rgb_sha256, 64)
                or type(self.width) is not int or type(self.height) is not int
                or not 0 < self.width <= MAX_SIDE or not 0 < self.height <= MAX_SIDE
                or self.width * self.height > MAX_PIXELS
                or type(self._pixels) is not bytes or len(self._pixels) != 3 * self.width * self.height
                or hashlib.sha256(self._pixels).hexdigest() != self.rgb_sha256
                or type(self._metadata) is not bytes or not isinstance(self.input_info()['metadata'], dict)):
            raise ProtocolError('input_proof_invalid')
        return self

    def rgb(self):
        return np.frombuffer(self._pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
