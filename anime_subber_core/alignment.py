"""Global monotonic Gemini-to-Whisper alignment.

Unlike the old cursor-based matcher, this module scores the complete sequence. A
missed line is represented by a gap in the dynamic-programming path, so later
matches can recover and strong matches near the end act as reverse anchors.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .models import Subtitle
from .similarity import partial_ratio, ratio
from .text import normalize_text


@dataclass(frozen=True)
class Match:
    gemini_index: int
    whisper_start: int
    whisper_end: int
    score: float


def _candidate_score(gemini_text: str, segments: Sequence[dict], start: int, end: int) -> float:
    target = normalize_text(gemini_text)
    source = normalize_text("".join(str(s.get("text", "")) for s in segments[start : end + 1]))
    if not target or not source:
        return 0.0
    similarity = ratio(target, source) / 100.0
    partial = partial_ratio(target, source) / 100.0
    length_ratio = min(len(target), len(source)) / max(len(target), len(source))
    return (0.55 * similarity + 0.45 * partial) * (length_ratio ** 0.30)


def global_monotonic_matches(
    gemini_lines: Sequence[dict], whisper_segments: Sequence[dict], max_span: int = 5
) -> List[Match]:
    """Find a maximum-score monotonic path with skips on either side."""
    n, m = len(gemini_lines), len(whisper_segments)
    neg = float("-inf")
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    gemini_skip, whisper_skip = -0.24, -0.035

    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if current == neg:
                continue
            if i < n and current + gemini_skip > dp[i + 1][j]:
                dp[i + 1][j] = current + gemini_skip
                back[i + 1][j] = (i, j, None)
            if j < m and current + whisper_skip > dp[i][j + 1]:
                dp[i][j + 1] = current + whisper_skip
                back[i][j + 1] = (i, j, None)
            if i < n:
                ja = str(gemini_lines[i].get("ja", ""))
                for span in range(1, min(max_span, m - j) + 1):
                    score = _candidate_score(ja, whisper_segments, j, j + span - 1)
                    if score < 0.43:
                        continue
                    # A match must beat skipping the Gemini line; high-confidence
                    # matches become anchors from either end of the sequence.
                    reward = (score - 0.43) * 3.2 + 0.18 - 0.018 * (span - 1)
                    ni, nj = i + 1, j + span
                    if current + reward > dp[ni][nj]:
                        dp[ni][nj] = current + reward
                        back[ni][nj] = (i, j, Match(i, j, j + span - 1, score))

    i, j = n, m
    matches: List[Match] = []
    while i or j:
        step = back[i][j]
        if step is None:
            break
        pi, pj, match = step
        if match is not None:
            matches.append(match)
        i, j = pi, pj
    return list(reversed(matches))


def _spread_unmatched(
    indices: Iterable[int], lines: Sequence[dict], start: float, end: float
) -> List[Subtitle]:
    indexes = list(indices)
    if not indexes or end <= start:
        return []
    # Never extend a gap. Short gaps get compact, overlapping cues rather than
    # invented timestamps outside the media/chunk boundary.
    step = (end - start) / len(indexes)
    result = []
    for offset, index in enumerate(indexes):
        cue_start = start + offset * step
        cue_end = min(end, max(cue_start + 0.30, start + (offset + 1) * step - 0.03))
        result.append(Subtitle(cue_start, cue_end, str(lines[index].get("en", ""))))
    return result


def align_subtitles(
    gemini_lines: Sequence[dict],
    whisper_segments: Sequence[dict],
    chunk_start: float,
    chunk_end: float,
) -> Tuple[List[Subtitle], Set[int], List[Tuple[float, float, List[dict]]]]:
    valid = [g for g in gemini_lines if g.get("ja") and g.get("en") and "[NO SPEECH]" not in g.get("en", "")]
    matches = global_monotonic_matches(valid, whisper_segments)
    by_gemini: Dict[int, Match] = {match.gemini_index: match for match in matches}
    anchors = [(-1, chunk_start, chunk_start)]
    anchors.extend(
        (match.gemini_index, float(whisper_segments[match.whisper_start]["start"]),
         float(whisper_segments[match.whisper_end]["end"])) for match in matches
    )
    anchors.append((len(valid), chunk_end, chunk_end))

    cues: List[Subtitle] = []
    recovery: List[Tuple[float, float, List[dict]]] = []
    for left, right in zip(anchors, anchors[1:]):
        left_index, _, left_end = left
        right_index, right_start, _ = right
        missing = list(range(left_index + 1, right_index))
        if missing:
            recovery.append((max(chunk_start, left_end), min(chunk_end, right_start), [valid[k] for k in missing]))
            cues.extend(_spread_unmatched(missing, valid, max(chunk_start, left_end), min(chunk_end, right_start)))
        if right_index < len(valid):
            match = by_gemini[right_index]
            start = max(chunk_start, float(whisper_segments[match.whisper_start]["start"]))
            end = min(chunk_end, float(whisper_segments[match.whisper_end]["end"]))
            if end > start:
                cues.append(Subtitle(start, end, str(valid[right_index]["en"])))

    used = {
        int(whisper_segments[k].get("global_idx", k))
        for match in matches
        for k in range(match.whisper_start, match.whisper_end + 1)
    }
    return sorted(cues, key=lambda cue: cue.start), used, recovery
