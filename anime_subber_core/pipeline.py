import concurrent.futures
import shutil
from pathlib import Path

from .alignment import align_subtitles
from .cache import CacheStore
from .config import ORPHAN_GAP_THRESH_SEC, RuntimeConfig, SUPPORTED_EXTS, VIDEO_EXTS
from .gemini import (GeminiManager, submit_audio_chunks, submit_video_chunks,
                     translate_recovery_slice, translate_video_recovery_slice)
from .gemini_import import align_imported_dialogue, load_gemini_export
from .ocr import process_video_signs
from .layout import resolve_collisions
from .media import load_audio, prepare_gemini_video, video_resolution
from .similarity import ratio
from .subtitles import read_ass, read_srt, write_ass, write_srt
from .text import normalize_text
from .timing import refine_contiguous_starts
from .whisper_engine import (WhisperGate, load_model, safe_instance_count,
                             bound_segments, transcribe_full, transcribe_targeted)


def _missing_count(recovery):
    return sum(len(lines) for _, _, lines in recovery)


def _merge_recovery_segments(window, replacements, start, end):
    kept = [segment for segment in window
            if min(end, float(segment.get("end", 0))) - max(start, float(segment.get("start", 0))) <= 0.05]
    return sorted(kept + replacements, key=lambda segment: (segment.get("start", 0), segment.get("end", 0)))


def _bound_cues(cues, duration):
    """Clip cues to playable media and discard cues wholly inside decoder padding."""
    bounded = []
    for cue in cues:
        cue.start = max(0.0, cue.start)
        cue.end = min(duration, cue.end)
        if cue.start < duration and cue.end > cue.start:
            bounded.append(cue)
    return bounded


def _same_ocr_sign(old, cue, resolution=(1920, 1080)):
    overlap = min(old.end, cue.end) - max(old.start, cue.start)
    shorter = min(old.end - old.start, cue.end - cue.start)
    if overlap <= 0.05 or shorter <= 0 or overlap / shorter < 0.60:
        return False

    if old.effect and cue.effect and old.effect == cue.effect:
        return True

    x_tolerance = max(32.0, resolution[0] * 0.025)
    y_tolerance = max(32.0, resolution[1] * 0.040)
    if None not in (old.x, old.y, cue.x, cue.y):
        if abs(old.x - cue.x) > x_tolerance or abs(old.y - cue.y) > y_tolerance:
            return False
    elif old.position != cue.position:
        return False

    same_timing = (abs(old.start - cue.start) <= 0.40 and
                   abs(old.end - cue.end) <= 0.60)
    old_text, new_text = normalize_text(old.text), normalize_text(cue.text)
    text_similarity = ratio(old_text, new_text) if old_text and new_text else 0
    return same_timing or text_similarity >= 45


def _deduplicate_cues(cues, resolution=(1920, 1080)):
    """Remove overlap duplicates without collapsing distinct on-screen signs.

    Dialogue still requires exact text. OCR signs are matched primarily by
    timing/location because Gemini can translate the same Japanese differently
    on a later OCR-only refresh.
    """
    deduplicated = []

    for cue in sorted(cues, key=lambda item: (item.start, item.layer)):
        def is_duplicate(old):
            if old.layer != cue.layer:
                return False
            if cue.layer:
                return _same_ocr_sign(old, cue, resolution)
            return (old.text == cue.text and
                    min(old.end, cue.end) - max(old.start, cue.start) > 0.25)

        if not any(is_duplicate(old) for old in deduplicated):
            deduplicated.append(cue)
    return deduplicated


def process_video(video_file, output_path, run_ocr=False, ocr_only=False, use_lite=False,
                  runtime=RuntimeConfig(), whisper_model_name="large", device="cuda",
                  gemini_input=None):
    cache, manager = CacheStore(), GeminiManager(use_lite)
    cues, gemini_jobs, whisper_segments = [], [], []
    existing_ocr_cues, imported_ocr_cues = [], []
    imported_dialogue = []
    media_duration = None
    model, gate = None, None
    video_source = None
    is_video = Path(video_file).suffix.lower() in VIDEO_EXTS
    resolution = video_resolution(video_file) if is_video else (1920, 1080)
    if ocr_only and gemini_input:
        raise ValueError("--ocr-only and --gemini-input cannot be used together")
    if ocr_only and not is_video:
        raise ValueError("--ocr-only requires a video file; OCR is not available for audio-only inputs")
    if run_ocr and not is_video:
        print(f"[OCR] Skipping {video_file}: OCR requires a video input.")
        run_ocr = False

    if ocr_only and Path(output_path).exists():
        backup = str(cache.media_dir(video_file) / (Path(output_path).name + ".bak"))
        shutil.copy2(output_path, backup)
        print(f"Backed up existing subtitles to {backup}")
        if output_path.lower().endswith(".srt"):
            cues.extend(read_srt(output_path))
        elif output_path.lower().endswith(".ass"):
            cues.extend(read_ass(output_path))
        existing_ocr_cues = [cue for cue in cues if cue.layer]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, runtime.gemini_workers)) as pool:
        if not ocr_only:
            # Whisper still consumes 16 kHz mono audio locally. Gemini receives
            # the video itself for video inputs so visual context can improve
            # transcription/translation.
            audio = load_audio(video_file, cache)
            media_duration = len(audio) / 1000.0

            if gemini_input:
                imported_dialogue, imported_ocr_cues = load_gemini_export(
                    gemini_input, resolution
                )
                cues.extend(imported_ocr_cues)
                print(f"[Gemini Import] Loaded {len(imported_dialogue)} dialogue cue(s) and "
                      f"{len(imported_ocr_cues)} on-screen cue(s) from {gemini_input}")
            elif is_video:
                upload_path, mime_type = prepare_gemini_video(video_file, cache)
                if Path(upload_path).resolve() != Path(video_file).resolve():
                    print(f"[Gemini] Using cached compatible video proxy: {upload_path}")
                print(f"[Gemini] Uploading video for multimodal transcription: {Path(upload_path).name}")
                video_source = manager.upload_video(str(upload_path), mime_type)
                gemini_jobs = submit_video_chunks(
                    video_source, media_duration, video_file, manager, cache, pool
                )
            else:
                # Preserve explicit audio-file support.
                gemini_jobs = submit_audio_chunks(audio, video_file, manager, cache, pool)

            count = safe_instance_count(runtime.whisper_workers, whisper_model_name, device)
            gate = WhisperGate(count)
            cached_whisper = cache.load_json(video_file, "whisper_v2")
            if cached_whisper is not None:
                whisper_segments = bound_segments(cached_whisper, media_duration)
                for index, segment in enumerate(whisper_segments):
                    segment["global_idx"] = index
            else:
                if count != runtime.whisper_workers:
                    print(f"[Whisper] Capacity probe limited concurrency to {count} instance(s).")
                model = load_model(whisper_model_name, device)
                whisper_segments = transcribe_full(audio, video_file, model, cache, gate)

        # OCR starts only after the full Whisper pass releases the accelerator,
        # while outstanding Gemini transcription calls may continue.
        if (run_ocr or ocr_only) and not runtime.ocr_gpu:
            ocr_cues, resolution = process_video_signs(
                video_file, manager, cache, pool, runtime.ocr_gpu, runtime.ocr_vision_rescue
            )
            known_signs = existing_ocr_cues + imported_ocr_cues
            if known_signs:
                ocr_cues = [cue for cue in ocr_cues
                            if not any(_same_ocr_sign(old, cue, resolution)
                                       for old in known_signs)]
            cues.extend(ocr_cues)

        used_indices = set()
        if gemini_input:
            imported_cues, imported_used = align_imported_dialogue(
                imported_dialogue, whisper_segments, media_duration
            )
            cues.extend(imported_cues)
            used_indices.update(imported_used)
        else:
            for job in gemini_jobs:  # Preserve chunk order even though calls execute concurrently.
                data, start, end = job.result()
                window = [s for s in whisper_segments
                          if s.get("end", 0) >= start - 15 and s.get("start", 0) <= end + 15]
                chunk_cues, chunk_used, recovery = align_subtitles(data or [], window, start, end)
                severe = [(gap_start, gap_end, missing) for gap_start, gap_end, missing in recovery
                          if ((len(missing) >= 3 and gap_end - gap_start >= 8.0) or
                              gap_end - gap_start >= 30.0)]
                if severe:
                    if model is None:
                        print("[Whisper] Loading model for targeted recovery of hallucinated/unmatched spans...")
                        model = load_model(whisper_model_name, device)
                    recovered_window = list(window)
                    for gap_start, gap_end, missing in severe:
                        print(f"[Whisper] Re-transcribing {gap_start:.2f}-{gap_end:.2f}s in short windows "
                              f"for {len(missing)} unmatched line(s)...")
                        replacements = transcribe_targeted(
                            audio, video_file, gap_start, gap_end, model, cache, gate,
                            expected_lines=[line.get("ja", "") for line in missing]
                        )
                        recovered_window = _merge_recovery_segments(
                            recovered_window, replacements, gap_start, gap_end
                        )
                    retry_cues, retry_used, retry_recovery = align_subtitles(
                        data or [], recovered_window, start, end
                    )
                    if _missing_count(retry_recovery) < _missing_count(recovery):
                        print(f"[Alignment] Targeted Whisper recovery reduced unmatched lines from "
                              f"{_missing_count(recovery)} to {_missing_count(retry_recovery)}.")
                        chunk_cues, chunk_used, recovery = retry_cues, retry_used, retry_recovery
                    else:
                        print("[Alignment] Targeted Whisper recovery did not improve this chunk; "
                              "keeping the original alignment.")
                cues.extend(chunk_cues)
                used_indices.update(index for index in chunk_used if index >= 0)
                for gap_start, gap_end, missing in recovery:
                    print(f"[Alignment] Bounded {len(missing)} unmatched line(s) to "
                          f"{gap_start:.2f}-{gap_end:.2f}s")

            # Re-translate long unused Whisper regions only when the script is
            # doing its own Gemini transcription. A supplied Gem export is the
            # user's authoritative input and should not silently trigger extra
            # dialogue generation.
            orphan_blocks, current = [], []
            for segment in whisper_segments:
                duration = float(segment.get("end", 0)) - float(segment.get("start", 0))
                if (segment.get("global_idx") not in used_indices and duration <= 15.0 and
                        len(str(segment.get("text", "")).strip()) > 1):
                    current.append(segment)
                elif current:
                    orphan_blocks.append(current)
                    current = []
            if current:
                orphan_blocks.append(current)
            recovery_jobs = []
            for index, block in enumerate(orphan_blocks, 1):
                start = max(0.0, float(block[0]["start"]))
                end = min(media_duration, float(block[-1]["end"]))
                if end - start > ORPHAN_GAP_THRESH_SEC:
                    cache_name = f"gemini_recovery_{round(start * 1000)}_{round(end * 1000)}"
                    if video_source is not None:
                        future = pool.submit(
                            translate_video_recovery_slice, video_source, video_file, start, end,
                            cache_name, manager, cache
                        )
                    else:
                        future = pool.submit(
                            translate_recovery_slice, audio, video_file, start, end, cache_name,
                            manager, cache
                        )
                    recovery_jobs.append((block, future))
            for block, job in recovery_jobs:
                data, start, end = job.result()
                recovered, recovered_used, _ = align_subtitles(data or [], block, start, end)
                cues.extend(recovered)
                used_indices.update(recovered_used)

        # GPU OCR runs only after every full/targeted Whisper inference has
        # finished, preventing EasyOCR and Whisper from competing for VRAM.
        if (run_ocr or ocr_only) and runtime.ocr_gpu:
            ocr_cues, resolution = process_video_signs(
                video_file, manager, cache, pool, True, runtime.ocr_vision_rescue
            )
            known_signs = existing_ocr_cues + imported_ocr_cues
            if known_signs:
                ocr_cues = [cue for cue in ocr_cues
                            if not any(_same_ocr_sign(old, cue, resolution)
                                       for old in known_signs)]
            cues.extend(ocr_cues)

    if video_source is not None:
        manager.delete_uploaded_file(video_source)

    cues = _deduplicate_cues(cues, resolution)
    if media_duration is not None:
        cues = _bound_cues(cues, media_duration)
    if runtime.strict_timing and not ocr_only:
        refine_contiguous_starts(cues, audio)
    if output_path.lower().endswith(".ass"):
        resolve_collisions(cues, resolution)
        write_ass(output_path, cues, resolution)
    else:
        write_srt(output_path, cues)
    print(f"Saved {len(cues)} subtitle cues to {output_path}")


def process_target(target, output_format="ass", run_ocr=False, ocr_only=False, force_update=False,
                   use_lite=False, runtime=RuntimeConfig(), model="large", device="cuda",
                   gemini_input=None):
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if gemini_input and not path.is_file():
        raise ValueError("--gemini-input currently requires a single media file, not a directory")
    if gemini_input and not Path(gemini_input).is_file():
        raise FileNotFoundError(f"Gemini subtitle input does not exist: {gemini_input}")
    files = [path] if path.is_file() else sorted(
        (p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS),
        key=lambda item: str(item).casefold(),
    )
    if not files:
        print(f"No supported media files found under {path}")
        return
    for media in files:
        if media.suffix.lower() not in SUPPORTED_EXTS:
            continue
        output = media.with_suffix(f".{output_format}")
        if output.exists() and not (force_update or ocr_only):
            print(f"Skipping {media} ({output.name} already exists)")
        else:
            try:
                process_video(str(media), str(output), run_ocr, ocr_only, use_lite, runtime, model, device,
                              gemini_input)
            except Exception as exc:
                print(f"[Error] Failed to process {media}: {exc}")
                if path.is_file():
                    raise
