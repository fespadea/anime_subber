# anime_subber

Generate English subtitle tracks for Japanese media by combining Gemini transcription/translation with Whisper timing. The optional OCR pipeline detects and translates Japanese text rendered in the video, including vertical Japanese reconstructed top-to-bottom within columns and right-to-left across columns.

For video inputs, Gemini receives video context instead of the Whisper-only extracted WAV. You can also generate a structured subtitle draft manually with a Gemini Gem and pass that output directly to the script.

## Runtime requirements

- Python with the packages in `requirements.txt`.
- `ffmpeg` available on `PATH`.
- Google GenAI credentials configured for the `google-genai` SDK when using the built-in Gemini API path or OCR vision rescue.
- A CUDA-capable PyTorch installation is recommended for Whisper/EasyOCR GPU use. Whisper defaults to `--device cuda`; EasyOCR uses the CPU unless `--ocr-gpu` is supplied.

The project keeps heavyweight imports lazy, so `--help` and unit tests do not require the model packages to be installed.

## Basic usage

```bash
python anime_subber.py episode.mkv
python anime_subber.py episode.mkv --format ass --ocr
python anime_subber.py episode.mkv --no-ocr
python anime_subber.py episode.mkv --ocr-only
python anime_subber.py /path/to/episodes --force-update
```

ASS is the default and preferred output when OCR is enabled because it supports exact `\\pos(...)` placement. SRT can only approximate sign placement with alignment tags.

`--ocr-only` preserves cues already present in an existing project-generated ASS or SRT file, then adds/refines OCR cues. `--force-update` regenerates the output file but intentionally reuses valid model caches.

Gemini vision rescue for difficult vertical OCR regions is **off by default**. Enable it only when you want the extra Gemini vision calls:

```bash
python anime_subber.py episode.mkv --ocr-vision-rescue
```

## Using the Gemini Gem workflow

The repository includes two files for this workflow:

- `GEMINI_GEM_INSTRUCTIONS.md` — paste this into the custom instructions for a Gemini Gem.
- `GEMINI_GEM_GUIDE.md` — step-by-step setup and usage guide.

The short version is:

1. Create a Gemini Gem and paste the contents of `GEMINI_GEM_INSTRUCTIONS.md` into its instructions.
2. Upload your episode/video to that Gem.
3. Ask it to generate the anime-subber subtitle export.
4. Copy the returned fenced `json` code block into a file, for example `episode.gemini.json`.
5. Run:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --force-update
```

The script uses the Gem's Japanese dialogue and English translation, then runs Whisper locally to refine dialogue timing. On-screen text supplied by the Gem is imported directly as positioned sign cues.

If you want to trust the Gem's on-screen-text detection and skip local EasyOCR entirely:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --no-ocr --force-update
```

The importer accepts either raw JSON or a response saved with the surrounding Markdown ` ```json ` fences intact.

See [`GEMINI_GEM_GUIDE.md`](GEMINI_GEM_GUIDE.md) for the full workflow and [`GEMINI_GEM_INSTRUCTIONS.md`](GEMINI_GEM_INSTRUCTIONS.md) for the exact Gem prompt.

## Gemini API video behavior

When `--gemini-input` is not supplied, the normal API workflow sends video context to Gemini for video inputs. Gemini-supported containers are uploaded directly. Unsupported containers such as MKV are converted to a cached compatible MP4 proxy, preferring video stream-copy when possible so the picture is not unnecessarily re-encoded.

The local 16 kHz mono WAV is still generated because Whisper uses it for transcription/timing refinement. It is not the input sent to Gemini for video files.

## Caching

Intermediates are stored under `.cache/anime_subber/`. Cache identity incorporates the resolved media path, file size, and modification time so replacing a source file at the same path does not reuse stale transcription/OCR data. Prompt/schema-sensitive caches are versioned in code.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q anime_subber_core tests
python anime_subber.py --help
```

The unit suite is designed to run without Gemini, Whisper, EasyOCR, ffmpeg, or a GPU. Full end-to-end quality still needs validation on real media with the runtime dependencies installed.
