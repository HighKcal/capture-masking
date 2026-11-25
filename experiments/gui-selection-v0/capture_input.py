"""Detached RGB8 selection submitted by the screen capture UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

import numpy as np

from fresh_protocol import MAX_PIXELS, MAX_SIDE, canonical, hex_value

MAX_CAPTURE_METADATA_BYTES = 4096
_METADATA_KEYS = {"attempt_id", "screen_generation"}


class CaptureInputError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _validate_dimensions(width, height):
    if type(width) is not int or type(height) is not int:
        raise CaptureInputError("capture_input_invalid")
    if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE and width * height <= MAX_PIXELS):
        raise CaptureInputError("image_limit")


def _copy_metadata(value, attempt_id, screen_generation):
    if type(value) is not dict:
        raise CaptureInputError("capture_input_invalid")

    nodes = 256
    chars = MAX_CAPTURE_METADATA_BYTES

    def string(text):
        nonlocal chars
        if type(text) is not str or len(text) > chars:
            raise CaptureInputError("capture_input_invalid")
        chars -= len(text)
        return text

    def copy(item, depth):
        nonlocal nodes
        nodes -= 1
        if depth > 8 or nodes < 0:
            raise CaptureInputError("capture_input_invalid")
        kind = type(item)
        if kind is dict:
            return MappingProxyType({string(key): copy(child, depth + 1) for key, child in item.items()})
        if kind in (list, tuple):
            return tuple(copy(child, depth + 1) for child in item)
        if kind is str:
            return string(item)
        if item is None or kind is bool:
            return item
        if kind is int and -(2**63) <= item < 2**63:
            return item
        if kind is float and math.isfinite(item):
            return item
        raise CaptureInputError("capture_input_invalid")

    copied = copy(value, 0)
    if any(key in copied for key in _METADATA_KEYS):
        raise CaptureInputError("capture_input_invalid")
    try:
        length = len(canonical(dict(thaw_metadata(copied), attempt_id=attempt_id,
                                    screen_generation=screen_generation)))
    except ValueError:
        raise CaptureInputError("capture_input_invalid") from None
    if length > MAX_CAPTURE_METADATA_BYTES:
        raise CaptureInputError("capture_input_invalid")
    return copied


def thaw_metadata(value):
    """Copy frozen metadata to JSON-native types for a UI event."""
    if isinstance(value, Mapping):
        return {key: thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_metadata(item) for item in value]
    return value


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class CapturedInput:
    width: int
    height: int
    rgb_bytes: bytes = field(repr=False)
    attempt_id: str
    screen_generation: str
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.validate_rgb()
        if (type(self.attempt_id) is not str or not hex_value(self.attempt_id, 32)
                or type(self.screen_generation) is not str or not hex_value(self.screen_generation, 64)):
            raise CaptureInputError("capture_input_invalid")
        object.__setattr__(self, "metadata", _copy_metadata(
            self.metadata, self.attempt_id, self.screen_generation,
        ))

    def validate_rgb(self):
        _validate_dimensions(self.width, self.height)
        if type(self.rgb_bytes) is not bytes or len(self.rgb_bytes) != self.width * self.height * 3:
            raise CaptureInputError("capture_input_invalid")

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, attempt_id: str, screen_generation: str,
                 metadata: dict | None = None) -> CapturedInput:
        """Copy the selected H×W×3 uint8 pixels in row-major order."""
        if type(rgb) is not np.ndarray or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise CaptureInputError("capture_input_invalid")
        height, width, _ = rgb.shape
        _validate_dimensions(width, height)
        return cls(width, height, rgb.tobytes(order="C"), attempt_id, screen_generation,
                   {} if metadata is None else metadata)
