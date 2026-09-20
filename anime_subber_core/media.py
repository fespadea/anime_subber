"""Bounded, explicit media decoding helpers."""
import os
import subprocess

from .cache import CacheStore


AUDIO_EXTRACTION_TIMEOUT_SECONDS = 10 * 60


def _audio_command(video_file: str, output_file: str):
    return [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", video_file,
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        output_file,
    ]


def load_audio(video_file: str, cache: CacheStore):
    """Extract 16 kHz mono WAV to disk, with a hard timeout, then load it."""
    from pydub import AudioSegment

    destination = cache.path(video_file, "audio_16k_mono", suffix=".wav")
    if not destination.exists() or destination.stat().st_size < 44:
        partial = str(destination) + ".partial.wav"
        try:
            subprocess.run(
                _audio_command(video_file, partial),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=AUDIO_EXTRACTION_TIMEOUT_SECONDS,
                text=True,
            )
            if not os.path.exists(partial) or os.path.getsize(partial) < 44:
                raise RuntimeError("FFmpeg produced no usable audio")
            os.replace(partial, destination)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()
            message = detail[-1] if detail else "unknown FFmpeg error"
            raise RuntimeError(f"Audio extraction failed for {video_file}: {message}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Audio extraction exceeded {AUDIO_EXTRACTION_TIMEOUT_SECONDS // 60} minutes for {video_file}"
            ) from exc
        finally:
            if os.path.exists(partial):
                os.unlink(partial)
    return AudioSegment.from_wav(destination)
