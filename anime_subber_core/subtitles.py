import datetime
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from .models import Subtitle
from .styling import dialogue_font_size, ocr_font_size, resolution_scale


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
    dialogue_margin = round(max(16, 40 * scale))
    sign_margin = round(max(10, 20 * scale))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dialogue,Arial,{dialogue_size:.2f},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{outline:.2f},{shadow:.2f},2,{dialogue_margin},{dialogue_margin},{dialogue_margin},1
Style: Sign,Arial,{sign_size:.2f},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{outline:.2f},{shadow:.2f},5,{sign_margin},{sign_margin},{sign_margin},1

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
