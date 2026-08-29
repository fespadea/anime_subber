# Project guidance

## Purpose

This project generates English subtitle tracks by combining Gemini's Japanese transcription/translation with Whisper's timing, and can separately detect and translate on-screen Japanese text. `mkv_muxer.py` is a separate, stable utility.

## Structure

- `anime_subber.py`: backwards-compatible command-line entry point.
- `anime_subber_core/`: implementation package.
  - `alignment.py`: pure sequence alignment and bounded timestamp recovery.
  - `cache.py`: project-local cache paths and atomic JSON writes.
  - `gemini.py`, `whisper_engine.py`: model adapters.
  - `ocr.py`: on-screen text detection/translation.
  - `layout.py`: temporal/spatial collision resolution for ASS cues.
  - `subtitles.py`: SRT/ASS parsing and writing.
  - `pipeline.py`, `cli.py`: orchestration and CLI.
- `tests/`: tests that must not call Gemini, Whisper, ffmpeg, or a GPU.

## Development rules

- Keep model/network imports lazy so unit tests work without the large runtime dependencies.
- Keep all generated intermediates under `.cache/anime_subber/`, never beside source videos.
- A cache key must include the resolved media path so videos with the same filename do not collide.
- Whisper access is serialized through `WhisperGate`. Do not add a second concurrent Whisper inference unless a user explicitly opts in after a successful capacity probe.
- Gemini work may be parallelized, but preserve source chunk order when assembling subtitles.
- Alignment must be monotonic, globally scored, and bounded by real media/chunk times. Never manufacture timestamps past the audio duration.
- OCR placement belongs in ASS (`\\pos`); SRT position tags are compatibility-only approximations.
- Preserve the existing CLI where practical. Add new behavior behind flags when it could disrupt existing workflows.

## Verification

Run `python -m unittest discover -s tests -v` and `python anime_subber.py --help`. For alignment changes, include a regression test where an early missed line does not force later lines to the end of the video.
