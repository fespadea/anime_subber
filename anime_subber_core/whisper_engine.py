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


def transcribe_full(audio, media_path: str, model, cache: CacheStore, gate: WhisperGate):
    cached = cache.load_json(media_path, "whisper")
    if cached is not None:
        segments = cached
    else:
        handle, temporary = tempfile.mkstemp(suffix=".wav", dir=cache.media_dir(media_path))
        os.close(handle)
        try:
            audio.export(temporary, format="wav")
            with gate.inference():
                result = model.transcribe(temporary, language="ja")
            segments = result.get("segments", [])
            cache.save_json(media_path, "whisper", segments)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    for index, segment in enumerate(segments):
        segment["global_idx"] = index
    return segments
