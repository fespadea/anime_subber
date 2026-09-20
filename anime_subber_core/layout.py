"""Collision-aware, symmetric placement for ASS dialogue and OCR cues."""
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .models import Subtitle
from .styling import (dialogue_font_size, dialogue_margin,
                      minimum_ocr_font_size, ocr_font_size)


PADDING = 7.0


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def center_x(self):
        return (self.left + self.right) / 2

    @property
    def center_y(self):
        return (self.top + self.bottom) / 2

    def overlap_amount(self, other: "Rect", padding: float = PADDING):
        return (min(self.right, other.right) - max(self.left, other.left) + padding,
                min(self.bottom, other.bottom) - max(self.top, other.top) + padding)

    def overlaps(self, other: "Rect", padding: float = PADDING) -> bool:
        horizontal, vertical = self.overlap_amount(other, padding)
        return horizontal > 0 and vertical > 0


def _temporal_overlap(left: Subtitle, right: Subtitle) -> bool:
    return min(left.end, right.end) - max(left.start, right.start) > 0.05


def _dimensions(cue: Subtitle, font_size: float, resolution: Tuple[int, int]):
    lines = cue.text.splitlines() or [""]
    width = min(resolution[0] * 0.94, max(font_size * 2, max(len(line) for line in lines) * font_size * 0.54))
    height = max(font_size * 1.20, len(lines) * font_size * 1.18)
    return width, height


def _rect(cue: Subtitle, resolution: Tuple[int, int]) -> Rect:
    width, height = _dimensions(cue, cue.font_size or
                                (ocr_font_size(resolution) if cue.layer else dialogue_font_size(resolution)), resolution)
    if cue.layer:
        x = cue.x if cue.x is not None else resolution[0] / 2
        y = cue.y if cue.y is not None else resolution[1] / 2
    else:
        x = resolution[0] / 2
        y = resolution[1] - dialogue_margin(resolution) - height / 2
    return Rect(x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def _clamp(cue: Subtitle, resolution: Tuple[int, int]):
    width, height = _dimensions(cue, cue.font_size or ocr_font_size(resolution), resolution)
    cue.x = min(max(cue.x, 10 + width / 2), resolution[0] - 10 - width / 2)
    cue.y = min(max(cue.y, 10 + height / 2), resolution[1] - 10 - height / 2)


def _inside_screen(cue: Subtitle, resolution: Tuple[int, int], margin=10):
    rectangle = _rect(cue, resolution)
    epsilon = 1e-6
    return (rectangle.left >= margin - epsilon and rectangle.top >= margin - epsilon and
            rectangle.right <= resolution[0] - margin + epsilon and
            rectangle.bottom <= resolution[1] - margin + epsilon)


def _colliding_ocr_pairs(ocr, resolution):
    return [(left, right) for index, left in enumerate(ocr) for right in ocr[index + 1:]
            if _temporal_overlap(left, right) and _rect(left, resolution).overlaps(_rect(right, resolution))]


def _shrink_before_moving(ocr, dialogue, resolution):
    """Repeatedly shrink every collision participant before changing positions."""
    for cue in ocr:
        cue.font_size = cue.font_size or ocr_font_size(resolution)
    for _ in range(18):
        participants = set()
        for left, right in _colliding_ocr_pairs(ocr, resolution):
            participants.update((id(left), id(right)))
        for cue in ocr:
            if any(_temporal_overlap(cue, normal) and _rect(cue, resolution).overlaps(_rect(normal, resolution))
                   for normal in dialogue):
                participants.add(id(cue))
            if not _inside_screen(cue, resolution):
                participants.add(id(cue))
        minimum = minimum_ocr_font_size(resolution)
        shrinkable = [cue for cue in ocr if id(cue) in participants and cue.font_size > minimum]
        if not shrinkable:
            break
        for cue in shrinkable:
            cue.font_size = max(minimum, cue.font_size * 0.88)


def _direction(first: float, second: float, first_index: int, second_index: int):
    if abs(first - second) > 0.5:
        return -1 if first < second else 1
    return -1 if first_index < second_index else 1


def _separate_symmetrically(ocr, dialogue, resolution):
    """Move each OCR/OCR pair equally in opposite directions."""
    index_by_id = {id(cue): index for index, cue in enumerate(ocr)}
    for _ in range(100):
        deltas = {id(cue): [0.0, 0.0] for cue in ocr}
        collisions = 0
        for left, right in _colliding_ocr_pairs(ocr, resolution):
            collisions += 1
            left_rect, right_rect = _rect(left, resolution), _rect(right, resolution)
            horizontal, vertical = left_rect.overlap_amount(right_rect)
            if horizontal <= vertical:
                sign = _direction(left_rect.center_x, right_rect.center_x,
                                  index_by_id[id(left)], index_by_id[id(right)])
                distance = horizontal / 2 + 0.5
                deltas[id(left)][0] += sign * distance
                deltas[id(right)][0] -= sign * distance
            else:
                sign = _direction(left_rect.center_y, right_rect.center_y,
                                  index_by_id[id(left)], index_by_id[id(right)])
                distance = vertical / 2 + 0.5
                deltas[id(left)][1] += sign * distance
                deltas[id(right)][1] -= sign * distance

        # Dialogue is fixed; only the OCR cue moves, along the shortest axis and
        # directly away from the dialogue rectangle.
        for cue in ocr:
            cue_rect = _rect(cue, resolution)
            for normal in dialogue:
                normal_rect = _rect(normal, resolution)
                if not _temporal_overlap(cue, normal) or not cue_rect.overlaps(normal_rect):
                    continue
                collisions += 1
                horizontal, vertical = cue_rect.overlap_amount(normal_rect)
                if horizontal <= vertical:
                    deltas[id(cue)][0] += _direction(cue_rect.center_x, normal_rect.center_x, 0, 1) * (horizontal + 1)
                else:
                    deltas[id(cue)][1] += _direction(cue_rect.center_y, normal_rect.center_y, 0, 1) * (vertical + 1)
        if not collisions:
            break
        movement = 0.0
        for cue in ocr:
            dx, dy = deltas[id(cue)]
            # Multiple simultaneous pair forces are averaged to prevent a dense
            # credit screen from flinging one cue across the frame.
            cue.x += dx * 0.65
            cue.y += dy * 0.65
            _clamp(cue, resolution)
            movement += abs(dx) + abs(dy)
        if movement < 0.1:
            break


def resolve_collisions(cues: Iterable[Subtitle], resolution: Tuple[int, int]) -> List[Subtitle]:
    result = list(cues)
    ocr = [cue for cue in result if cue.layer]
    dialogue = [cue for cue in result if not cue.layer]
    for cue in ocr:
        cue.x = cue.x if cue.x is not None else resolution[0] / 2
        cue.y = cue.y if cue.y is not None else resolution[1] / 2
    _shrink_before_moving(ocr, dialogue, resolution)
    for cue in ocr:
        _clamp(cue, resolution)
    _separate_symmetrically(ocr, dialogue, resolution)
    for cue in ocr:
        _clamp(cue, resolution)
    return result
