from dataclasses import dataclass


VIDEO_EXTS = (".mp4", ".mkv", ".mpeg", ".mpg", ".avi", ".mov", ".flv", ".webm", ".wmv", ".3gp")
AUDIO_EXTS = (".mp3", ".wav")
SUPPORTED_EXTS = VIDEO_EXTS + AUDIO_EXTS
CHUNK_LENGTH_MS = 4 * 60 * 1000
OVERLAP_MS = 30 * 1000
STEP_MS = CHUNK_LENGTH_MS - OVERLAP_MS
ORPHAN_GAP_THRESH_SEC = 20.0


@dataclass(frozen=True)
class RuntimeConfig:
    gemini_workers: int = 4
    whisper_workers: int = 1
    ocr_gpu: bool = False
    strict_timing: bool = False
    ocr_vision_rescue: bool = False
