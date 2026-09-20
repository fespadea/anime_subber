"""Import structured subtitle drafts produced by a Gemini Gem or chat."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from .alignment import global_monotonic_matches
from .models import Subtitle
from .text import parse_llm_json


GEM_EXPORT_FORMAT = "anime-subber-gem-v1"

_POSITION_MAP = {
    "bottom_left": 1,
    "bottom_center": 2,
    "bottom_right": 3,
    "middle_left": 4,
    "middle_center": 5,
    "middle_right": 6,
    "top_left": 7,
    "top_center": 8,
    "top_right": 9,
    "center": 5,
}


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"invalid timestamp: {value!r}")


def _position(value):
    if value is None:
        return None
    if isinstance(value, int) and 1 <= value <= 9:
        return value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text.isdigit() and 1 <= int(text) <= 9:
        return int(text)
    return _POSITION_MAP.get(text)


def _fingerprint(item: dict) -> str:
    raw = "|".join((
        str(item.get("ja", "")).strip(),
        f"{float(item.get('_start', 0.0)):.2f}",
        f"{float(item.get('_end', 0.0)):.2f}",
        str(item.get("position", "")),
    ))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_gemini_export(path: str, resolution=(1920, 1080)):
    """Load ``anime-subber-gem-v1`` JSON.

    Returns ``(dialogue, signs)``. Dialogue entries retain rough timestamps as
    ``_start``/``_end`` so Whisper can refine them. Screen text is converted
    directly to positioned subtitle cues.
    """
    raw_text = Path(path).read_text(encoding="utf-8-sig")
    payload = parse_llm_json(raw_text)
    if payload is None:
        raise ValueError("Gemini subtitle input is not valid JSON")
    if isinstance(payload, dict):
        format_name = payload.get("format")
        if format_name not in (None, GEM_EXPORT_FORMAT):
            raise ValueError(f"Unsupported Gemini subtitle format: {format_name!r}")
        raw_cues = payload.get("cues")
    else:
        raw_cues = payload
    if not isinstance(raw_cues, list):
        raise ValueError("Gemini subtitle input must be a JSON object with a 'cues' array or a bare array")

    width, height = resolution
    dialogue = []
    signs = []
    for index, item in enumerate(raw_cues):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "dialogue")).strip().lower().replace("-", "_")
        try:
            start, end = _seconds(item.get("start")), _seconds(item.get("end"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Gemini cue {index} has invalid start/end timestamps") from exc
        if end <= start:
            raise ValueError(f"Gemini cue {index} ends before it starts")
        ja = str(item.get("ja", "")).strip()
        en = str(item.get("en", "")).strip()
        if not en:
            continue

        if kind in {"screen_text", "screen", "sign", "ocr", "on_screen", "onscreen"}:
            position = _position(item.get("position")) or 5
            x = item.get("x")
            y = item.get("y")
            try:
                x = float(x) if x is not None else None
                y = float(y) if y is not None else None
            except (TypeError, ValueError):
                x = y = None
            # Allow normalized 0..1 coordinates when a Gem chooses to provide
            # more precision than the coarse 3x3 position label.
            if x is not None and y is not None and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x, y = x * width, y * height
            elif x is not None and y is not None and not (0 <= x <= width and 0 <= y <= height):
                x = y = None
            source = {**item, "_start": start, "_end": end, "position": position}
            signs.append(Subtitle(
                start, end, f"[{en}]", position=position, x=x, y=y, layer=1,
                effect=f"anime_subber_gem:{_fingerprint(source)}",
            ))
        else:
            if not ja:
                # Japanese source is what allows Whisper to refine timing. If
                # Gemini omitted it, retain the cue and trust Gemini's timing.
                ja = ""
            dialogue.append({"ja": ja, "en": en, "_start": start, "_end": end})

    dialogue.sort(key=lambda item: (item["_start"], item["_end"]))
    signs.sort(key=lambda cue: (cue.start, cue.end))
    return dialogue, signs


def align_imported_dialogue(lines: Sequence[dict], whisper_segments: Sequence[dict],
                            media_duration: float, padding: float = 4.0):
    """Refine Gem-provided dialogue timestamps against Whisper when possible.

    The Gem timestamp remains the fallback. Each line is aligned only inside a
    small time neighborhood, avoiding accidental matches to repeated short
    phrases elsewhere in the episode.
    """
    cues = []
    used = set()
    for line in lines:
        start = max(0.0, float(line["_start"]))
        end = min(media_duration, float(line["_end"]))
        if end <= start:
            continue
        english = str(line.get("en", "")).strip()
        japanese = str(line.get("ja", "")).strip()
        if not japanese:
            cues.append(Subtitle(start, end, english))
            continue

        left, right = max(0.0, start - padding), min(media_duration, end + padding)
        window = [segment for segment in whisper_segments
                  if float(segment.get("end", 0)) >= left and float(segment.get("start", 0)) <= right]
        matches = global_monotonic_matches([{"ja": japanese, "en": english}], window)
        if matches:
            match = matches[0]
            refined_start = max(0.0, match.start_time)
            refined_end = min(media_duration, match.end_time)
            # A narrow search window is already protective, but keep obviously
            # pathological matches from replacing a reasonable Gem timestamp.
            if (refined_end > refined_start and
                    refined_start <= end + padding and refined_end >= start - padding):
                cues.append(Subtitle(refined_start, refined_end, english))
                for index in range(match.whisper_start, match.whisper_end + 1):
                    if 0 <= index < len(window):
                        used.add(int(window[index].get("global_idx", index)))
                continue
        cues.append(Subtitle(start, end, english))
    return cues, used
