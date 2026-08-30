import os
import tempfile
import threading
import math
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


def _transcribe(model, audio_path, initial_prompt=None, force_speech=False):
    options = {
        "language": "ja",
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2.0,
    }
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    if force_speech:
        # Gemini has independently confirmed dialogue/lyrics in this targeted
        # span. Do not let Whisper's music-biased no-speech heuristics discard it.
        options["no_speech_threshold"] = None
        options["logprob_threshold"] = None
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
                        cache: CacheStore, gate: WhisperGate, expected_lines=None,
                        window_seconds=25.0):
    """Re-transcribe a bad span in short, context-reset, serialized windows."""
    start = max(0.0, start)
    end = min(len(audio) / 1000.0, end)
    import hashlib
    lines = [str(line) for line in (expected_lines or []) if line]
    prompt = "。".join(lines)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10] if prompt else "none"
    cache_name = f"whisper_recovery_v3_{round(start * 1000)}_{round(end * 1000)}_{prompt_hash}"
    cached = cache.load_json(media_path, cache_name)
    if cached is not None:
        return cached
    recovered = []
    cursor = start
    window_count = max(1, math.ceil((end - start) / window_seconds))
    window_index = 0
    while cursor < end - 0.05:
        window_end = min(end, cursor + window_seconds)
        handle, temporary = tempfile.mkstemp(suffix=".wav", dir=cache.media_dir(media_path))
        os.close(handle)
        try:
            audio[round(cursor * 1000):round(window_end * 1000)].export(temporary, format="wav")
            if lines and len(lines) >= window_count:
                line_start = max(0, math.floor(window_index * len(lines) / window_count) - 1)
                line_end = min(len(lines), math.ceil((window_index + 1) * len(lines) / window_count) + 1)
                window_prompt = "。".join(lines[line_start:line_end])
            else:
                window_prompt = prompt
            with gate.inference():
                result = _transcribe(model, temporary, initial_prompt=window_prompt or None,
                                     force_speech=bool(window_prompt))
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
        window_index += 1
    cache.save_json(media_path, cache_name, recovered)
    return recovered
