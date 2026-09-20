# Project guidance

## Purpose

This project generates English subtitle tracks by combining Gemini's Japanese transcription/translation with Whisper timing, and can separately detect and translate on-screen Japanese text. For video inputs, Gemini receives video (not the Whisper-only extracted WAV) so visual context can improve transcription. Structured subtitles created manually in a Gemini Gem/chat can also be imported directly. `mkv_muxer.py` is a separate, stable utility.

## Structure

- `anime_subber.py`: backwards-compatible command-line entry point.
- `anime_subber_core/`: implementation package.
  - `alignment.py`: pure sequence alignment and bounded timestamp recovery.
  - `cache.py`: project-local cache paths and atomic JSON writes.
  - `gemini.py`, `whisper_engine.py`: model adapters.
  - `gemini_import.py`: validation/import of `anime-subber-gem-v1` Gem/chat JSON and Whisper timing refinement.
  - `media.py`: Whisper audio extraction and Gemini-compatible video preparation.
  - `ocr.py`: on-screen text detection/translation, including vertical Japanese reconstruction.
  - `layout.py`: temporal/spatial collision resolution for ASS cues.
  - `subtitles.py`: SRT/ASS parsing and writing.
  - `pipeline.py`, `cli.py`: orchestration and CLI.
- `GEMINI_GEM_INSTRUCTIONS.md`: exact custom instructions/schema for a Gemini Gem that can produce direct script input.
- `GEMINI_GEM_GUIDE.md`: end-user setup and usage guide for the Gem workflow.
- `README.md`: installation, normal CLI usage, and an overview of the Gem workflow.
- `tests/`: tests that must not call Gemini, Whisper, ffmpeg, or a GPU.

## Development rules

- Keep model/network imports lazy so unit tests work without the large runtime dependencies.
- Keep all generated intermediates under `.cache/anime_subber/`, never beside source videos.
- Cache identity must include the resolved media path and source-file state so same-named files do not collide and replaced media does not reuse stale results.
- Whisper access is serialized through `WhisperGate`. Do not add a second concurrent Whisper inference unless a user explicitly opts in after a successful capacity probe.
- Whisper word timestamps should be preferred for character timing when they sufficiently cover and agree with the segment transcription; retain segment-level interpolation as a compatibility fallback.
- For video inputs, upload one Gemini-supported video and use clipped video metadata for chunk/recovery requests. Do not send extracted audio to Gemini when video is available. Unsupported containers such as MKV should use a cached compatible MP4 proxy; prefer stream-copying video before transcoding it.
- The 16 kHz mono WAV remains necessary for local Whisper and strict-timing analysis; "Gemini uses video" does not remove Whisper's local audio extraction.
- Gemini work may be parallelized, but preserve source chunk order when assembling subtitles.
- `--gemini-input` imports `anime-subber-gem-v1` JSON. Treat imported dialogue timestamps as hints and refine them against Whisper when the Japanese source matches; fall back to the supplied timestamps when it does not. Imported on-screen cues are authoritative sign candidates and should suppress duplicate OCR detections by timing/location.
- Alignment must be monotonic, globally scored, bounded by real media/chunk times, and honor its configured maximum Whisper-segment span. Never manufacture timestamps past the media duration.
- Vertical Japanese OCR is read top-to-bottom within a column and right-to-left across adjacent columns. Keep weak EasyOCR boxes long enough for geometry-first vertical rescue; likely tategaki crops may be verified/transcribed with Gemini vision before translation. Do not reorder tategaki into left-to-right column order.
- OCR placement belongs in ASS (`\\pos`); SRT position tags are compatibility-only approximations.
- `--ocr-only` must preserve existing cues in either project-generated SRT or ASS output before adding refreshed OCR cues. Existing ASS OCR signs take precedence over re-translations matched by timing/location; project-generated OCR cues should retain their `anime_subber_ocr:*` Effect provenance when round-tripped.
- Gemini vision rescue for likely vertical OCR regions is **off by default** because it is comparatively expensive and rarely needed. Enable it explicitly with `--ocr-vision-rescue`.
- Preserve the existing CLI where practical. Add new behavior behind flags when it could disrupt existing workflows.

## Verification

Run `python -m unittest discover -s tests -v` and `python anime_subber.py --help`. For alignment changes, include regressions showing that an early missed line does not force later lines to the end of the video and that usable Whisper word timestamps preserve local pauses. For Gem-import changes, test both Whisper-refined and timestamp-fallback behavior. For Gemini-video changes, unit-test request/proxy construction without network calls and manually smoke-test at least one unsupported container through the proxy path when possible.
