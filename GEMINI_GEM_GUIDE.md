# Using a Gemini Gem with anime_subber

This workflow lets Gemini watch the whole video and produce a structured subtitle draft that `anime_subber` can consume directly. It is useful when Gemini's full-video visual understanding gives better results than the script's normal chunked Gemini/OCR workflow, especially for stylized or vertical Japanese text.

The Gem output is **not** meant to be an ASS or SRT file. It uses the `anime-subber-gem-v1` JSON format so the script can still refine dialogue timing with Whisper and place on-screen text intelligently.

## 1. Create the Gem

1. Open Gemini and create a new custom Gem.
2. Give it a name such as **Anime Subber**.
3. Open `GEMINI_GEM_INSTRUCTIONS.md` from this repository.
4. Copy the entire contents into the Gem's instruction field.
5. Save the Gem.

Those instructions tell Gemini to return exactly one fenced `json` code block and no surrounding commentary, which makes the result easy to copy without formatting damage.

## 2. Generate subtitles in Gemini

Upload the video to the Gem and send a short request such as:

```text
Generate the anime-subber subtitle export for this video.
```

The Gem should return one JSON code block with data similar to:

```json
{
  "format": "anime-subber-gem-v1",
  "cues": [
    {
      "kind": "dialogue",
      "start": 12.345,
      "end": 15.210,
      "ja": "そういうことじゃないよ",
      "en": "That's not what I mean."
    },
    {
      "kind": "screen_text",
      "start": 18.100,
      "end": 22.750,
      "ja": "ありがとう",
      "en": "Thank you",
      "position": "top_right",
      "x": 0.82,
      "y": 0.30,
      "orientation": "vertical"
    }
  ]
}
```

For vertical Japanese, the Gem is instructed to read top-to-bottom within a column and right-to-left across adjacent columns.

## 3. Save the Gem output

Use Gemini's code-block copy button and save the copied JSON to a local file, for example:

```text
episode.gemini.json
```

You can also save the response with its surrounding Markdown code fences intact. `anime_subber` accepts a single fenced `json` block as well as plain JSON.

Do not manually convert the Gem output to SRT or ASS first. Keeping the structured JSON preserves Japanese source text, cue type, orientation, and positioning data that the script can use.

## 4. Run anime_subber with the Gem output

Run:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --force-update
```

By default this will:

- load dialogue and on-screen text from the Gem export;
- run Whisper locally on the episode audio;
- use the Gem's Japanese dialogue text to refine approximate dialogue timestamps against Whisper;
- fall back to Gemini's supplied timestamps if a dialogue line cannot be aligned safely;
- import Gem on-screen text directly as positioned sign cues;
- still run local EasyOCR and add signs that the Gem may have missed;
- suppress local OCR signs that match Gem-provided signs by timing/location;
- write ASS output by default.

`--gemini-input` replaces the script's normal Gemini API dialogue-generation step. It does **not** disable Whisper because Whisper is still useful for precise local timing.

## 5. Choose whether to keep local OCR enabled

Local OCR remains enabled by default:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --force-update
```

This is useful if you want EasyOCR to supplement Gemini's screen-text results.

If Gemini already found the written text well and you want to avoid extra OCR work or possible redundant sign detections, disable local OCR:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --no-ocr --force-update
```

For videos with difficult vertical text, the Gem workflow is often a better primary visual-text source than local OCR because Gemini can use the entire scene as context.

## 6. OCR vision rescue is opt-in

Gemini vision rescue for difficult EasyOCR vertical regions is intentionally off by default. Normal OCR does not trigger those additional image-model calls unless you explicitly request them:

```bash
python anime_subber.py "episode.mkv" --ocr-vision-rescue
```

You can combine it with a Gem import, although this is usually unnecessary if the Gem already handled the video's written text well:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --ocr-vision-rescue --force-update
```

## 7. Output format

ASS is the default and is recommended when the export contains `screen_text` because it supports accurate positioning:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --format ass --force-update
```

SRT is also available:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --format srt --force-update
```

SRT cannot represent precise arbitrary screen coordinates as reliably as ASS, so ASS is preferred when on-screen text matters.

## 8. Updating an existing subtitle file

If the output file already exists, normal processing skips it unless `--force-update` is supplied. Use:

```bash
python anime_subber.py "episode.mkv" --gemini-input "episode.gemini.json" --force-update
```

`--ocr-only` cannot be combined with `--gemini-input`. The Gem workflow represents a complete subtitle-source pass, while `--ocr-only` is specifically for augmenting an already existing subtitle file with OCR.

## 9. Batch processing

`--gemini-input` is intentionally restricted to one media file because one Gem export corresponds to one specific video. For a directory of episodes, create one Gem export per episode and run the command separately for each pair.

Example:

```bash
python anime_subber.py "Episode 01.mkv" --gemini-input "Episode 01.gemini.json" --force-update
python anime_subber.py "Episode 02.mkv" --gemini-input "Episode 02.gemini.json" --force-update
```

## 10. Troubleshooting

### The script says the Gemini input is invalid JSON

Make sure the file contains either:

- the raw JSON object; or
- exactly one Markdown code block beginning with ` ```json ` and ending with ` ``` `.

Do not include extra conversational text around the block.

### Dialogue timing looks a little different from Gemini's timestamps

That is expected. Gemini's timestamps are treated as approximate hints. When the Japanese source text can be matched safely, Whisper refines the timing to the actual speech.

### A Gem dialogue line could not be matched by Whisper

The script keeps Gemini's original start/end times rather than forcing a bad alignment.

### Written text is duplicated

Gem-provided `screen_text` cues suppress matching EasyOCR signs based primarily on timing and location. If you still see duplicates and the Gem already captured the signs reliably, run with `--no-ocr`.

### I only want to use the script's normal Gemini API workflow

Do not pass `--gemini-input`:

```bash
python anime_subber.py "episode.mkv" --force-update
```

For video files, the script sends video context to Gemini rather than audio-only input. MKV and other unsupported containers are converted to a cached compatible video proxy as needed.
