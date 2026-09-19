import concurrent.futures
import shutil
from pathlib import Path

from .alignment import align_subtitles
from .cache import CacheStore
from .config import ORPHAN_GAP_THRESH_SEC, RuntimeConfig, SUPPORTED_EXTS
from .gemini import GeminiManager, submit_audio_chunks, translate_recovery_slice
from .ocr import process_video_signs
from .layout import resolve_collisions
from .media import load_audio
from .subtitles import read_srt, write_ass, write_srt
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


def process_video(video_file, output_path, run_ocr=False, ocr_only=False, use_lite=False,
                  runtime=RuntimeConfig(), whisper_model_name="large", device="cuda"):
    cache, manager = CacheStore(), GeminiManager(use_lite)
    cues, resolution, gemini_jobs, whisper_segments = [], (1920, 1080), [], []
    media_duration = None
    model, gate = None, None
    if ocr_only and Path(output_path).exists():
        backup = str(cache.media_dir(video_file) / (Path(output_path).name + ".bak"))
        shutil.copy2(output_path, backup)
        print(f"Backed up existing subtitles to {backup}")
        if output_path.lower().endswith(".srt"):
            cues.extend(read_srt(output_path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, runtime.gemini_workers)) as pool:
        if not ocr_only:
            audio = load_audio(video_file, cache)
            media_duration = len(audio) / 1000.0
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
                # Gemini calls run while this single full-track Whisper pass executes.
                whisper_segments = transcribe_full(audio, video_file, model, cache, WhisperGate(count))
        # OCR starts only after Whisper releases the accelerator, while outstanding Gemini calls continue.
        if (run_ocr or ocr_only) and not runtime.ocr_gpu:
            ocr_cues, resolution = process_video_signs(video_file, manager, cache, pool, runtime.ocr_gpu)
            cues.extend(ocr_cues)
        used_indices = set()
        for job in gemini_jobs:  # Preserve chunk order even though calls execute concurrently.
            data, start, end = job.result()
            window = [s for s in whisper_segments if s.get("end", 0) >= start - 15 and s.get("start", 0) <= end + 15]
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
                    replacements = transcribe_targeted(audio, video_file, gap_start, gap_end,
                                                       model, cache, gate,
                                                       expected_lines=[line.get("ja", "") for line in missing])
                    recovered_window = _merge_recovery_segments(recovered_window, replacements,
                                                                gap_start, gap_end)
                retry_cues, retry_used, retry_recovery = align_subtitles(data or [], recovered_window, start, end)
                if _missing_count(retry_recovery) < _missing_count(recovery):
                    print(f"[Alignment] Targeted Whisper recovery reduced unmatched lines from "
                          f"{_missing_count(recovery)} to {_missing_count(retry_recovery)}.")
                    chunk_cues, chunk_used, recovery = retry_cues, retry_used, retry_recovery
                else:
                    print("[Alignment] Targeted Whisper recovery did not improve this chunk; keeping the original alignment.")
            cues.extend(chunk_cues)
            used_indices.update(index for index in chunk_used if index >= 0)
            for gap_start, gap_end, missing in recovery:
                print(f"[Alignment] Bounded {len(missing)} unmatched line(s) to {gap_start:.2f}-{gap_end:.2f}s")
        # Re-translate long unused Whisper regions. Calls are submitted together,
        # so independent recovery regions use the same Gemini worker pool.
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
                recovery_jobs.append((block, pool.submit(translate_recovery_slice, audio, video_file, start, end,
                                                        f"gemini_recovery_{round(start * 1000)}_{round(end * 1000)}",
                                                        manager, cache)))
        for block, job in recovery_jobs:
            data, start, end = job.result()
            recovered, recovered_used, _ = align_subtitles(data or [], block, start, end)
            cues.extend(recovered)
            used_indices.update(recovered_used)
        # GPU OCR runs only after every full/targeted Whisper inference has
        # finished, preventing EasyOCR and Whisper from competing for VRAM.
        if (run_ocr or ocr_only) and runtime.ocr_gpu:
            ocr_cues, resolution = process_video_signs(video_file, manager, cache, pool, True)
            cues.extend(ocr_cues)
    # Overlapping Gemini chunks can repeat the same utterance. Keep the best
    # first cue when normalized text and timing substantially overlap.
    deduplicated = []
    for cue in sorted(cues, key=lambda item: (item.start, item.layer)):
        duplicate = next((old for old in deduplicated if old.text == cue.text and
                          min(old.end, cue.end) - max(old.start, cue.start) > 0.25), None)
        if duplicate is None:
            deduplicated.append(cue)
    cues = deduplicated
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


def process_target(target, output_format="srt", run_ocr=False, ocr_only=False, force_update=False,
                   use_lite=False, runtime=RuntimeConfig(), model="large", device="cuda"):
    path = Path(target)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    for media in files:
        if media.suffix.lower() not in SUPPORTED_EXTS:
            continue
        output = media.with_suffix(f".{output_format}")
        if output.exists() and not (force_update or ocr_only):
            print(f"Skipping {media} ({output.name} already exists)")
        else:
            try:
                process_video(str(media), str(output), run_ocr, ocr_only, use_lite, runtime, model, device)
            except Exception as exc:
                print(f"[Error] Failed to process {media}: {exc}")
                if path.is_file():
                    raise
