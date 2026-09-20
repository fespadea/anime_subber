import concurrent.futures
import io
import threading
import time
from typing import Sequence

from .cache import CacheStore
from .config import CHUNK_LENGTH_MS, STEP_MS
from .text import parse_llm_json


_AUDIO_CACHE_VERSION = "v2"


def _audio_lines(value):
    """Validate Gemini audio JSON before it reaches the aligner."""
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ja = str(item.get("ja", "")).strip()
        en = str(item.get("en", "")).strip()
        if ja and en:
            lines.append({"ja": ja, "en": en})
    return lines


def _translation_items(value):
    """Validate OCR translation JSON and normalize numeric IDs."""
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            identifier = int(item["id"])
        except (TypeError, ValueError):
            continue
        english = str(item.get("en", "")).strip()
        if english:
            items.append({"id": identifier, "en": english})
    return items


class GeminiManager:
    def __init__(self, use_lite: bool = False, client_factory=None):
        # Keep aliases first so normal runs follow Google's supported Flash
        # track, with explicit stable fallbacks in case an alias is unavailable.
        self.models = (["gemini-flash-lite-latest", "gemini-3.5-flash-lite"] if use_lite else
                       ["gemini-flash-latest", "gemini-3.8-flash",
                        "gemini-flash-lite-latest", "gemini-3.5-flash-lite"])
        self.api_exhausted = False
        self._lock = threading.Lock()
        self._client = None
        self._client_factory = client_factory

    @property
    def client(self):
        # Gemini requests start concurrently, so lazy initialization must be
        # atomic. Otherwise two clients can be created and the SDK finalizer can
        # close the displaced client while another thread is still using it.
        with self._lock:
            if self._client is None:
                if self._client_factory is None:
                    from google import genai
                    self._client = genai.Client()
                else:
                    self._client = self._client_factory()
            return self._client

    def _discard_client(self, client):
        with self._lock:
            if self._client is client:
                self._client = None

    def generate(self, contents, config, prefer_lite: bool = False):
        with self._lock:
            available = list(self.models)
        if prefer_lite:
            available.sort(key=lambda model: "lite" not in model)
        for model in available:
            for attempt in range(2):
                active_client = self.client
                try:
                    return active_client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                except Exception as exc:
                    message = str(exc).lower()
                    if "client has been closed" in message and attempt == 0:
                        self._discard_client(active_client)
                        print("[Gemini] Client was closed unexpectedly; recreating it and retrying once...")
                        continue
                    if any(word in message for word in ("429", "quota", "exhausted")):
                        if attempt == 0:
                            # A burst of parallel chunk requests can hit a
                            # transient rate limit. Retry once before declaring
                            # this model unusable for the remainder of the run.
                            time.sleep(1.0)
                            continue
                        with self._lock:
                            if model in self.models:
                                self.models.remove(model)
                            self.api_exhausted = not self.models
                    else:
                        print(f"[Gemini] {model} failed: {exc}")
                    break
        return None


def make_audio_config():
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction=(
            "You are an expert anime translator. Transcribe the spoken Japanese and translate it into natural "
            "English. Ignore music and effects. Return only a JSON array of objects with 'ja' and 'en' keys."
        ),
        temperature=0.1,
        response_mime_type="application/json",
    )


def _translate_chunk(audio, media_path: str, index: int, start_ms: int, end_ms: int,
                     manager: GeminiManager, cache: CacheStore, config):
    name = f"gemini_audio_{_AUDIO_CACHE_VERSION}_chunk_{index + 1}"
    cached = cache.load_json(media_path, name)
    if cached is not None:
        return _audio_lines(cached), start_ms / 1000.0, end_ms / 1000.0
    if manager.api_exhausted:
        return [], start_ms / 1000.0, end_ms / 1000.0
    from google.genai import types
    buffer = io.BytesIO()
    audio[start_ms:end_ms].export(buffer, format="wav")
    response = manager.generate(
        [types.Part.from_bytes(data=buffer.getvalue(), mime_type="audio/wav"),
         "Transcribe and translate this audio into the requested JSON format."], config
    )
    parsed = parse_llm_json(response.text) if response else None
    data = _audio_lines(parsed)
    # An empty JSON array is a valid "no speech" result and should be cached;
    # malformed/non-array output should be retried on a later run instead.
    if isinstance(parsed, list):
        cache.save_json(media_path, name, data)
    return data, start_ms / 1000.0, end_ms / 1000.0


def submit_audio_chunks(audio, media_path: str, manager: GeminiManager, cache: CacheStore,
                        executor: concurrent.futures.Executor, config=None):
    config = config or make_audio_config()
    jobs = []
    for index, start_ms in enumerate(range(0, len(audio), STEP_MS)):
        end_ms = min(start_ms + CHUNK_LENGTH_MS, len(audio))
        jobs.append(executor.submit(_translate_chunk, audio, media_path, index, start_ms, end_ms,
                                    manager, cache, config))
        if end_ms >= len(audio):
            break
    return jobs


def translate_text_batch(items: Sequence[dict], manager: GeminiManager, prefer_lite: bool = True):
    from google.genai import types
    import json
    prompt = (
        "Translate the Japanese on-screen text to concise English. Newlines in Japanese input may separate "
        "vertical columns that have already been ordered in Japanese reading order (rightmost column first). "
        "Return only a JSON array of objects with the exact input 'id' and an 'en' key.\n" +
        json.dumps(items, ensure_ascii=False)
    )
    response = manager.generate([prompt], types.GenerateContentConfig(temperature=0.1,
                                response_mime_type="application/json"), prefer_lite=prefer_lite)
    return _translation_items(parse_llm_json(response.text)) if response else []


def translate_recovery_slice(audio, media_path: str, start: float, end: float, cache_name: str,
                             manager: GeminiManager, cache: CacheStore, config=None):
    cache_name = f"gemini_audio_{_AUDIO_CACHE_VERSION}_{cache_name}"
    cached = cache.load_json(media_path, cache_name)
    if cached is not None:
        return _audio_lines(cached), start, end
    if manager.api_exhausted:
        return [], start, end
    from google.genai import types
    config = config or make_audio_config()
    buffer = io.BytesIO()
    audio[max(0, round(start * 1000) - 500):min(len(audio), round(end * 1000) + 500)].export(buffer, format="wav")
    response = manager.generate(
        [types.Part.from_bytes(data=buffer.getvalue(), mime_type="audio/wav"),
         "Transcribe and translate this missed audio segment into the requested JSON format."], config
    )
    parsed = parse_llm_json(response.text) if response else None
    data = _audio_lines(parsed)
    if isinstance(parsed, list):
        cache.save_json(media_path, cache_name, data)
    return data, start, end
