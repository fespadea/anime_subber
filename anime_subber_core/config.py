from dataclasses import dataclass


SUPPORTED_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav")
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
