"""Bounded, explicit media decoding and Gemini-video preparation helpers."""
import os
import subprocess
from pathlib import Path

from .cache import CacheStore


AUDIO_EXTRACTION_TIMEOUT_SECONDS = 10 * 60
VIDEO_PROXY_TIMEOUT_SECONDS = 30 * 60

GEMINI_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
}


def _audio_command(video_file: str, output_file: str):
    return [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", video_file,
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        output_file,
    ]


def _gemini_remux_command(video_file: str, output_file: str):
    """Keep the source video stream when possible and normalize audio to AAC."""
    return [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", video_file,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_file,
    ]


def _gemini_transcode_command(video_file: str, output_file: str):
    """Universal fallback when the source video codec cannot be muxed into MP4."""
    return [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", video_file,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_file,
    ]


def _run_ffmpeg(command, timeout_seconds: int, description: str):
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout_seconds,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        message = detail[-1] if detail else "unknown FFmpeg error"
        raise RuntimeError(f"{description} failed: {message}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{description} exceeded {timeout_seconds // 60} minutes") from exc


def load_audio(video_file: str, cache: CacheStore):
    """Extract 16 kHz mono WAV for Whisper, with a hard timeout, then load it."""
    from pydub import AudioSegment

    destination = cache.path(video_file, "audio_16k_mono", suffix=".wav")
    if not destination.exists() or destination.stat().st_size < 44:
        partial = str(destination) + ".partial.wav"
        try:
            _run_ffmpeg(_audio_command(video_file, partial), AUDIO_EXTRACTION_TIMEOUT_SECONDS,
                        f"Audio extraction for {video_file}")
            if not os.path.exists(partial) or os.path.getsize(partial) < 44:
                raise RuntimeError("FFmpeg produced no usable audio")
            os.replace(partial, destination)
        finally:
            if os.path.exists(partial):
                os.unlink(partial)
    return AudioSegment.from_wav(destination)


def prepare_gemini_video(video_file: str, cache: CacheStore):
    """Return a Gemini-supported local video path and MIME type.

    Supported containers are uploaded directly. Containers such as Matroska are
    converted to a cached MP4 proxy. The first attempt keeps the original video
    stream and only normalizes audio; a full H.264 transcode is a fallback.
    """
    source = Path(video_file)
    mime_type = GEMINI_VIDEO_MIME_TYPES.get(source.suffix.lower())
    if mime_type:
        return source, mime_type

    destination = cache.path(video_file, "gemini_video_v1", suffix=".mp4")
    if destination.exists() and destination.stat().st_size > 0:
        return destination, "video/mp4"

    partial = Path(str(destination) + ".partial.mp4")
    partial.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            _run_ffmpeg(_gemini_remux_command(video_file, str(partial)),
                        VIDEO_PROXY_TIMEOUT_SECONDS,
                        f"Gemini video remux for {video_file}")
        except RuntimeError as remux_error:
            if partial.exists():
                partial.unlink()
            print(f"[Gemini] Fast video remux failed; transcoding a compatible MP4 instead: {remux_error}")
            _run_ffmpeg(_gemini_transcode_command(video_file, str(partial)),
                        VIDEO_PROXY_TIMEOUT_SECONDS,
                        f"Gemini video transcode for {video_file}")
        if not partial.exists() or partial.stat().st_size <= 0:
            raise RuntimeError("FFmpeg produced no usable Gemini video proxy")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination, "video/mp4"


def video_resolution(video_file: str, fallback=(1920, 1080)):
    """Read frame dimensions with ffprobe without opening a full decoder."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_file],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            check=True, timeout=30,
        )
        width_text, height_text = result.stdout.strip().split("x", 1)
        width, height = int(width_text), int(height_text)
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return fallback
