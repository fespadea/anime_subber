"""Resolution-relative ASS style measurements."""
from typing import Tuple


REFERENCE_HEIGHT = 1080.0
REFERENCE_DIALOGUE_FONT = 42.0
REFERENCE_OCR_FONT = 42.0

# User-adjustable global subtitle size. For example, 1.25 is 25% larger and
# 0.90 is 10% smaller. This affects both dialogue and OCR subtitle fonts.
SUBTITLE_SIZE_SCALE = 1.25


def resolution_scale(resolution: Tuple[int, int]) -> float:
    return max(0.1, resolution[1] / REFERENCE_HEIGHT)


def dialogue_font_size(resolution: Tuple[int, int]) -> float:
    return max(15.0 * SUBTITLE_SIZE_SCALE,
               REFERENCE_DIALOGUE_FONT * resolution_scale(resolution) * SUBTITLE_SIZE_SCALE)


def ocr_font_size(resolution: Tuple[int, int]) -> float:
    return max(16.0 * SUBTITLE_SIZE_SCALE,
               REFERENCE_OCR_FONT * resolution_scale(resolution) * SUBTITLE_SIZE_SCALE)


def minimum_ocr_font_size(resolution: Tuple[int, int]) -> float:
    return max(10.0 * SUBTITLE_SIZE_SCALE,
               14.0 * resolution_scale(resolution) * SUBTITLE_SIZE_SCALE)
