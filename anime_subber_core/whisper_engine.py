import os
import tempfile
import threading
from contextlib import contextmanager

from .cache import CacheStore


class WhisperGate:
    """Serializes GPU inference, including any future targeted recovery passes."""
    def __init__(self, max_instances: int = 1):
        if max_instances < 1:
            raise ValueError("max_instances must be positive")
        self.max_instances = max_instances
        self._semaphore = threading.BoundedSemaphore(max_instances)

    @contextmanager
    def inference(self):
        with self._semaphore:
            yield


def safe_instance_count(requested: int, model_name: str, device: str) -> int:
    """Conservative capacity probe; multiple instances require explicit request and ample free VRAM."""
    if requested <= 1 or device != "cuda":
        return 1
    try:
        import torch
        free_bytes, _ = torch.cuda.mem_get_info()
        # Large Whisper commonly needs several GB plus working memory. Keep a
        # conservative 6 GiB allowance per concurrent instance.
        return max(1, min(requested, int(free_bytes // (6 * 1024 ** 3))))
    except Exception:
        return 1


def load_model(model_name: str = "large", device: str = "cuda"):
    import whisper
    return whisper.load_model(model_name, device=device)


def _transcribe(model, audio_path):
    options = {
        "language": "ja",
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2.0,
    }
    try:
        return model.transcribe(audio_path, **options)
    except TypeError:
        # Compatibility with older openai-whisper versions.
        options.pop("hallucination_silence_threshold")
        return model.transcribe(audio_path, **options)


def transcribe_full(audio, media_path: str, model, cache: CacheStore, gate: WhisperGate):
    cached = cache.load_json(media_path, "whisper_v2")
    if cached is not None:
        segments = cached
    else:
        handle, temporary = tempfile.mkstemp(suffix=".wav", dir=cache.media_dir(media_path))
        os.close(handle)
        try:
            audio.export(temporary, format="wav")
            with gate.inference():
                result = _transcribe(model, temporary)
            segments = result.get("segments", [])
            cache.save_json(media_path, "whisper_v2", segments)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    for index, segment in enumerate(segments):
        segment["global_idx"] = index
    return segments


def transcribe_targeted(audio, media_path: str, start: float, end: float, model,
                        cache: CacheStore, gate: WhisperGate, window_seconds=25.0):
    """Re-transcribe a bad span in short, context-reset, serialized windows."""
    start = max(0.0, start)
    end = min(len(audio) / 1000.0, end)
    cache_name = f"whisper_recovery_v2_{round(start * 1000)}_{round(end * 1000)}"
    cached = cache.load_json(media_path, cache_name)
    if cached is not None:
        return cached
    recovered = []
    cursor = start
    while cursor < end - 0.05:
        window_end = min(end, cursor + window_seconds)
        handle, temporary = tempfile.mkstemp(suffix=".wav", dir=cache.media_dir(media_path))
        os.close(handle)
        try:
            audio[round(cursor * 1000):round(window_end * 1000)].export(temporary, format="wav")
            with gate.inference():
                result = _transcribe(model, temporary)
            for segment in result.get("segments", []):
                adjusted = dict(segment)
                adjusted["start"] = max(start, cursor + float(segment.get("start", 0)))
                adjusted["end"] = min(end, cursor + float(segment.get("end", 0)))
                adjusted["global_idx"] = -1
                if adjusted["end"] > adjusted["start"]:
                    recovered.append(adjusted)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        cursor = window_end
    cache.save_json(media_path, cache_name, recovered)
    return recovered
