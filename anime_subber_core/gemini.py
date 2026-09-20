import concurrent.futures
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .cache import CacheStore
from .config import CHUNK_LENGTH_MS, STEP_MS
from .text import parse_llm_json


_AUDIO_CACHE_VERSION = "v3"
_VIDEO_CACHE_VERSION = "v1"
_VIDEO_UPLOAD_TIMEOUT_SECONDS = 10 * 60


@dataclass(frozen=True)
class GeminiVideoSource:
    name: str
    uri: str
    mime_type: str


def _audio_lines(value):
    """Validate Gemini transcription JSON before it reaches the aligner."""
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


def _vision_items(value):
    """Validate Gemini image-OCR JSON and normalize numeric IDs."""
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
        japanese = str(item.get("ja", "")).strip()
        if japanese:
            items.append({"id": identifier, "ja": japanese})
    return items


class GeminiManager:
    def __init__(self, use_lite: bool = False, client_factory=None):
        self.models = (["gemini-flash-lite-latest", "gemini-3.5-flash-lite"] if use_lite else
                       ["gemini-flash-latest", "gemini-3.8-flash",
                        "gemini-flash-lite-latest", "gemini-3.5-flash-lite"])
        self.api_exhausted = False
        self._lock = threading.Lock()
        self._client = None
        self._client_factory = client_factory

    @property
    def client(self):
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

    def upload_video(self, path: str, mime_type: str,
                     timeout_seconds: int = _VIDEO_UPLOAD_TIMEOUT_SECONDS):
        """Upload one video and wait for Gemini's processing step to finish."""
        from google.genai import types

        uploaded = self.client.files.upload(
            file=str(path), config=types.UploadFileConfig(mime_type=mime_type)
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", str(state or "")).upper()
            if state_name.endswith("ACTIVE"):
                break
            if state_name.endswith("FAILED"):
                raise RuntimeError(f"Gemini failed to process uploaded video {Path(path).name}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Gemini video processing timed out for {Path(path).name}")
            time.sleep(2.0)
            uploaded = self.client.files.get(name=uploaded.name)
        return GeminiVideoSource(
            name=str(uploaded.name), uri=str(uploaded.uri),
            mime_type=str(getattr(uploaded, "mime_type", None) or mime_type),
        )

    def delete_uploaded_file(self, source: GeminiVideoSource):
        try:
            self.client.files.delete(name=source.name)
        except Exception as exc:
            print(f"[Gemini] Could not delete temporary uploaded video {source.name}: {exc}")


def make_transcription_config():
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction=(
            "You are an expert Japanese-to-English anime subtitle translator. Transcribe only spoken Japanese "
            "and translate it into natural concise English subtitles. Use the video frames as context to resolve "
            "names, speakers, objects, jokes, ambiguous words, and scene context. Do not create entries solely "
            "for written on-screen text unless that text is actually spoken. Ignore music and sound effects. "
            "Preserve utterance order. Return only a JSON array of objects with 'ja' and 'en' keys."
        ),
        temperature=0.1,
        response_mime_type="application/json",
    )


# Backwards-compatible name used by older callers/tests.
def make_audio_config():
    return make_transcription_config()


def _translate_audio_chunk(audio, media_path: str, index: int, start_ms: int, end_ms: int,
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
         "Transcribe and translate the spoken Japanese in this audio into the requested JSON format."], config
    )
    parsed = parse_llm_json(response.text) if response else None
    data = _audio_lines(parsed)
    if isinstance(parsed, list):
        cache.save_json(media_path, name, data)
    return data, start_ms / 1000.0, end_ms / 1000.0


def submit_audio_chunks(audio, media_path: str, manager: GeminiManager, cache: CacheStore,
                        executor: concurrent.futures.Executor, config=None):
    """Audio-only fallback for explicit audio inputs."""
    config = config or make_transcription_config()
    jobs = []
    for index, start_ms in enumerate(range(0, len(audio), STEP_MS)):
        end_ms = min(start_ms + CHUNK_LENGTH_MS, len(audio))
        jobs.append(executor.submit(_translate_audio_chunk, audio, media_path, index, start_ms, end_ms,
                                    manager, cache, config))
        if end_ms >= len(audio):
            break
    return jobs


def _video_part(source: GeminiVideoSource, start: float, end: float):
    from google.genai import types
    return types.Part(
        file_data=types.FileData(file_uri=source.uri, mime_type=source.mime_type),
        video_metadata=types.VideoMetadata(
            start_offset=f"{max(0.0, start):.3f}s",
            end_offset=f"{max(start, end):.3f}s",
            fps=1.0,
        ),
    )


def _translate_video_chunk(source: GeminiVideoSource, media_path: str, index: int,
                           start: float, end: float, manager: GeminiManager,
                           cache: CacheStore, config):
    name = f"gemini_video_{_VIDEO_CACHE_VERSION}_chunk_{index + 1}"
    cached = cache.load_json(media_path, name)
    if cached is not None:
        return _audio_lines(cached), start, end
    if manager.api_exhausted:
        return [], start, end
    response = manager.generate(
        [_video_part(source, start, end),
         "Transcribe and translate only the spoken Japanese in this video interval. Use the visible scene as "
         "context, but do not turn unrelated written text into dialogue."],
        config,
    )
    parsed = parse_llm_json(response.text) if response else None
    data = _audio_lines(parsed)
    if isinstance(parsed, list):
        cache.save_json(media_path, name, data)
    return data, start, end


def submit_video_chunks(source: GeminiVideoSource, duration: float, media_path: str,
                        manager: GeminiManager, cache: CacheStore,
                        executor: concurrent.futures.Executor, config=None):
    """Submit overlapping clipped intervals from one uploaded video."""
    config = config or make_transcription_config()
    jobs = []
    step = STEP_MS / 1000.0
    chunk = CHUNK_LENGTH_MS / 1000.0
    index = 0
    start = 0.0
    while start < duration:
        end = min(start + chunk, duration)
        jobs.append(executor.submit(_translate_video_chunk, source, media_path, index, start, end,
                                    manager, cache, config))
        if end >= duration:
            break
        index += 1
        start += step
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


def recognize_japanese_image_batch(items: Sequence[dict], manager: GeminiManager,
                                   prefer_lite: bool = False):
    if not items:
        return []
    from google.genai import types

    contents = [(
        "Transcribe the Japanese text in each cropped image. These crops are "
        "suspected vertical Japanese (tategaki), although some may be false "
        "positives. Read characters top-to-bottom within a vertical column and "
        "read adjacent columns from right to left. Preserve separate vertical "
        "columns with newline characters in that reading order. Do not translate, "
        "romanize, or explain the text. If an image has no Japanese text, omit it "
        "from the result. Return only a JSON array of objects with the exact input "
        "'id' and a 'ja' key."
    )]
    for item in items:
        contents.append(f"Image id {int(item['id'])}:")
        contents.append(types.Part.from_bytes(data=item["image_bytes"], mime_type="image/jpeg"))

    response = manager.generate(
        contents,
        types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        prefer_lite=prefer_lite,
    )
    return _vision_items(parse_llm_json(response.text)) if response else []


def translate_recovery_slice(audio, media_path: str, start: float, end: float, cache_name: str,
                             manager: GeminiManager, cache: CacheStore, config=None):
    """Audio-only recovery for explicit audio inputs."""
    cache_name = f"gemini_audio_{_AUDIO_CACHE_VERSION}_{cache_name}"
    cached = cache.load_json(media_path, cache_name)
    if cached is not None:
        return _audio_lines(cached), start, end
    if manager.api_exhausted:
        return [], start, end
    from google.genai import types
    config = config or make_transcription_config()
    buffer = io.BytesIO()
    audio[max(0, round(start * 1000) - 500):min(len(audio), round(end * 1000) + 500)].export(
        buffer, format="wav"
    )
    response = manager.generate(
        [types.Part.from_bytes(data=buffer.getvalue(), mime_type="audio/wav"),
         "Transcribe and translate the spoken Japanese in this missed audio interval."], config
    )
    parsed = parse_llm_json(response.text) if response else None
    data = _audio_lines(parsed)
    if isinstance(parsed, list):
        cache.save_json(media_path, cache_name, data)
    return data, start, end


def translate_video_recovery_slice(source: GeminiVideoSource, media_path: str, start: float, end: float,
                                   cache_name: str, manager: GeminiManager, cache: CacheStore,
                                   config=None):
    cache_name = f"gemini_video_{_VIDEO_CACHE_VERSION}_{cache_name}"
    cached = cache.load_json(media_path, cache_name)
    if cached is not None:
        return _audio_lines(cached), start, end
    if manager.api_exhausted:
        return [], start, end
    config = config or make_transcription_config()
    clipped_start, clipped_end = max(0.0, start - 0.5), end + 0.5
    response = manager.generate(
        [_video_part(source, clipped_start, clipped_end),
         "Transcribe and translate the spoken Japanese in this missed video interval. Use the frames for "
         "context and ignore unrelated written text."],
        config,
    )
    parsed = parse_llm_json(response.text) if response else None
    data = _audio_lines(parsed)
    if isinstance(parsed, list):
        cache.save_json(media_path, cache_name, data)
    return data, start, end
