"""Consumption-aware, global monotonic Gemini-to-Whisper alignment."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .models import Subtitle
from .japanese import phonetic_text
from .similarity import partial_ratio, ratio
from .text import normalize_text


@dataclass(frozen=True)
class _TimedCharacter:
    value: str
    segment: int
    start: float
    end: float


@dataclass(frozen=True)
class Match:
    gemini_index: int
    whisper_start: int
    whisper_end: int
    score: float
    character_start: int = 0
    character_end: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


def _character_timeline(segments: Sequence[dict]) -> List[_TimedCharacter]:
    timeline = []
    for segment_index, segment in enumerate(segments):
        normalized = normalize_text(str(segment.get("text", "")))
        if not normalized:
            continue
        start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
        duration = max(0.05, end - start)
        for offset, character in enumerate(normalized):
            timeline.append(_TimedCharacter(character, segment_index,
                                            start + duration * offset / len(normalized),
                                            start + duration * (offset + 1) / len(normalized)))
    return timeline


def _orthographic_score(target: str, source: str) -> float:
    if not target or not source:
        return 0.0
    similarity = ratio(target, source) / 100.0
    partial = partial_ratio(target, source) / 100.0
    length_ratio = min(len(target), len(source)) / max(len(target), len(source))
    return (0.60 * similarity + 0.40 * partial) * (length_ratio ** 0.35)


def _score(target: str, source: str) -> float:
    orthographic = _orthographic_score(target, source)
    target_reading, source_reading = phonetic_text(target), phonetic_text(source)
    if not target_reading or not source_reading:
        return orthographic
    return max(orthographic, _orthographic_score(target_reading, source_reading) * 0.97)


def _length_options(target_length: int, remaining: int):
    values = set(range(max(1, target_length - 2), target_length + 3))
    values.update(round(target_length * factor) for factor in (0.6, 0.75, 1.25, 1.5, 1.8))
    return sorted(value for value in values if 1 <= value <= remaining)


def _candidates(index: int, line: dict, timeline: Sequence[_TimedCharacter], limit=120):
    target = normalize_text(str(line.get("ja", "")))
    if not target:
        return []
    text = "".join(character.value for character in timeline)
    candidates = []
    for start in range(len(timeline)):
        for length in _length_options(len(target), len(timeline) - start):
            end = start + length
            score = _score(target, text[start:end])
            crossed_boundaries = timeline[end - 1].segment - timeline[start].segment
            if crossed_boundaries:
                score -= 0.08 * crossed_boundaries
            if score >= 0.43:
                candidates.append(Match(index, timeline[start].segment, timeline[end - 1].segment, score,
                                        start, end, timeline[start].start, timeline[end - 1].end))
    candidates.sort(key=lambda match: (-match.score, match.character_start))
    return candidates[:limit]


def global_monotonic_matches(gemini_lines: Sequence[dict], whisper_segments: Sequence[dict],
                             max_span: int = 5) -> List[Match]:
    """Align non-overlapping character ranges, including ranges in one segment."""
    timeline = _character_timeline(whisper_segments)
    states = [(0, 0.0, tuple())]
    for index, line in enumerate(gemini_lines):
        candidates = _candidates(index, line, timeline)
        next_by_end = {}
        for consumed, total, path in states:
            skipped = (consumed, total - 0.24, path)
            if consumed not in next_by_end or skipped[1] > next_by_end[consumed][1]:
                next_by_end[consumed] = skipped
            for match in candidates:
                if match.character_start < consumed:
                    continue
                gap_cost = min(0.30, (match.character_start - consumed) * 0.002)
                reward = (match.score - 0.43) * 3.2 + 0.18 - gap_cost
                state = (match.character_end, total + reward, path + (match,))
                if match.character_end not in next_by_end or state[1] > next_by_end[match.character_end][1]:
                    next_by_end[match.character_end] = state
        states = sorted(next_by_end.values(), key=lambda state: state[1], reverse=True)[:240]
    return list(max(states, key=lambda state: state[1])[2]) if states else []


def _spread_unmatched(indices: Iterable[int], lines: Sequence[dict], start: float, end: float):
    indexes = list(indices)
    if not indexes or end <= start:
        return []
    step = (end - start) / len(indexes)
    return [Subtitle(start + offset * step,
                     min(end, max(start + offset * step + 0.30, start + (offset + 1) * step - 0.03)),
                     str(lines[index].get("en", ""))) for offset, index in enumerate(indexes)]


def align_subtitles(gemini_lines: Sequence[dict], whisper_segments: Sequence[dict],
                    chunk_start: float, chunk_end: float):
    valid = [g for g in gemini_lines if g.get("ja") and g.get("en") and "[NO SPEECH]" not in g.get("en", "")]
    matches = global_monotonic_matches(valid, whisper_segments)
    by_gemini: Dict[int, Match] = {match.gemini_index: match for match in matches}
    anchors = [(-1, chunk_start, chunk_start)]
    anchors.extend((match.gemini_index, match.start_time, match.end_time) for match in matches)
    anchors.append((len(valid), chunk_end, chunk_end))
    cues, recovery = [], []
    for left, right in zip(anchors, anchors[1:]):
        left_index, _, left_end = left
        right_index, right_start, _ = right
        missing = list(range(left_index + 1, right_index))
        if missing:
            bounded_start, bounded_end = max(chunk_start, left_end), min(chunk_end, right_start)
            recovery.append((bounded_start, bounded_end, [valid[k] for k in missing]))
            cues.extend(_spread_unmatched(missing, valid, bounded_start, bounded_end))
        if right_index < len(valid):
            match = by_gemini[right_index]
            start, end = max(chunk_start, match.start_time), min(chunk_end, match.end_time)
            if end > start:
                cues.append(Subtitle(start, end, str(valid[right_index]["en"])))
    used = {int(whisper_segments[k].get("global_idx", k)) for match in matches
            for k in range(match.whisper_start, match.whisper_end + 1)}
    return sorted(cues, key=lambda cue: cue.start), used, recovery
