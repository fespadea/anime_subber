import re
from pathlib import Path
from typing import Iterable, Tuple

from .models import Subtitle
from .styling import (dialogue_font_size, dialogue_margin, ocr_font_size,
                      resolution_scale, sign_margin)


def read_srt(path: str):
    text = Path(path).read_text(encoding="utf-8-sig")
    cues = []
    pattern = re.compile(r"(?ms)^\d+\s*\n(\d\d:\d\d:\d\d[,.]\d{3})\s+-->\s+(\d\d:\d\d:\d\d[,.]\d{3})\s*\n(.*?)(?=\n\s*\n\d+\s*\n|\Z)")
    def seconds(value):
        hours, minutes, rest = value.replace(",", ".").split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(rest)
    for start, end, body in pattern.findall(text):
        position = None
        match = re.match(r"\{\\an([1-9])\}", body)
        if match:
            position = int(match.group(1))
            body = body[match.end():]
        cues.append(Subtitle(seconds(start), seconds(end), body.strip(), position))
    return cues


def _ass_seconds(value: str) -> float:
    hours, minutes, rest = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _ass_unescape(text: str) -> str:
    """Reverse the small ASS escape subset emitted by :func:`write_ass`."""
    output = []
    index = 0
    while index < len(text):
        if text[index] != "\\" or index + 1 >= len(text):
            output.append(text[index])
            index += 1
            continue
        escaped = text[index + 1]
        if escaped == "N":
            output.append("\n")
        elif escaped in ("\\", "{", "}"):
            output.append(escaped)
        else:
            # Preserve unknown ASS escapes rather than silently corrupting text.
            output.extend(("\\", escaped))
        index += 2
    return "".join(output)


def read_ass(path: str):
    """Read cues from ASS files written by this project.

    This intentionally parses only the event fields and override tags that
    ``write_ass`` emits. Unknown tags remain in the visible text rather than
    being interpreted incorrectly.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    cues = []
    for raw_line in text.splitlines():
        if not raw_line.startswith("Dialogue:"):
            continue
        fields = raw_line[len("Dialogue:"):].lstrip().split(",", 9)
        if len(fields) != 10:
            continue
        layer_text, start_text, end_text, _style, _name, _ml, _mr, _mv, _effect, body = fields
        try:
            layer = int(layer_text.strip())
            start, end = _ass_seconds(start_text), _ass_seconds(end_text)
        except (TypeError, ValueError):
            continue

        position = None
        x = y = font_size = None
        override = re.match(r"^\{([^}]*)\}", body)
        if override:
            tags = override.group(1)
            pos = re.search(r"\\pos\((-?[0-9.]+),(-?[0-9.]+)\)", tags)
            anchor = re.search(r"\\an([1-9])", tags)
            size = re.search(r"\\fs([0-9.]+)", tags)
            if pos:
                x, y = float(pos.group(1)), float(pos.group(2))
            if anchor:
                position = int(anchor.group(1))
            if size:
                font_size = float(size.group(1))
            # Only strip an override group when every tag in it is one we emit.
            remainder = re.sub(r"\\pos\(-?[0-9.]+,-?[0-9.]+\)|\\an[1-9]|\\fs[0-9.]+", "", tags)
            if not remainder:
                body = body[override.end():]
        cues.append(Subtitle(start, end, _ass_unescape(body), position, x, y, layer, font_size))
    return cues


def _srt_time(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _ass_time(seconds: float) -> str:
    centiseconds = round(max(0.0, seconds) * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def write_srt(path: str, cues: Iterable[Subtitle]):
    blocks = []
    for index, cue in enumerate(sorted(cues, key=lambda item: item.start), 1):
        text = cue.text
        if cue.position:
            text = f"{{\\an{cue.position}}}{text}"
        blocks.append(f"{index}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{text}\n")
    Path(path).write_text("\n".join(blocks), encoding="utf-8")


def write_ass(path: str, cues: Iterable[Subtitle], resolution: Tuple[int, int] = (1920, 1080)):
    width, height = resolution
    scale = resolution_scale(resolution)
    dialogue_size = dialogue_font_size(resolution)
    sign_size = ocr_font_size(resolution)
    outline = max(1.2, 2.5 * scale)
    shadow = max(0.5, 1.0 * scale)
    dialogue_margin_px = dialogue_margin(resolution)
    sign_margin_px = sign_margin(resolution)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dialogue,Arial,{dialogue_size:.2f},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{outline:.2f},{shadow:.2f},2,{dialogue_margin_px},{dialogue_margin_px},{dialogue_margin_px},1
Style: Sign,Arial,{sign_size:.2f},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{outline:.2f},{shadow:.2f},5,{sign_margin_px},{sign_margin_px},{sign_margin_px},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for cue in sorted(cues, key=lambda item: item.start):
        style = "Sign" if cue.position or cue.x is not None else "Dialogue"
        override = ""
        if cue.x is not None and cue.y is not None:
            override = f"{{\\pos({round(cue.x)},{round(cue.y)})}}"
        elif cue.position:
            override = f"{{\\an{cue.position}}}"
        if cue.font_size is not None:
            override = override[:-1] + f"\\fs{round(cue.font_size)}}}" if override else f"{{\\fs{round(cue.font_size)}}}"
        events.append(f"Dialogue: {cue.layer},{_ass_time(cue.start)},{_ass_time(cue.end)},{style},,0,0,0,,{override}{_ass_escape(cue.text)}")
    Path(path).write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
