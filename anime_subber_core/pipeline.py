import concurrent.futures
import shutil
from pathlib import Path

from .alignment import align_subtitles
from .cache import CacheStore
from .config import ORPHAN_GAP_THRESH_SEC, RuntimeConfig, SUPPORTED_EXTS
from .gemini import GeminiManager, submit_audio_chunks, translate_recovery_slice
from .ocr import process_video_signs
from .subtitles import read_srt, write_ass, write_srt
from .timing import refine_contiguous_starts
from .whisper_engine import WhisperGate, load_model, safe_instance_count, transcribe_full


def process_video(video_file, output_path, run_ocr=False, ocr_only=False, use_lite=False,
                  runtime=RuntimeConfig(), whisper_model_name="large", device="cuda"):
    from pydub import AudioSegment
    cache, manager = CacheStore(), GeminiManager(use_lite)
    cues, resolution, gemini_jobs, whisper_segments = [], (1920, 1080), [], []
    if ocr_only and output_path.lower().endswith(".srt") and Path(output_path).exists():
        backup = output_path + ".bak"
        shutil.copy2(output_path, backup)
        cues.extend(read_srt(output_path))
        print(f"Backed up existing subtitles to {backup}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, runtime.gemini_workers)) as pool:
        if not ocr_only:
            audio = AudioSegment.from_file(video_file)
            gemini_jobs = submit_audio_chunks(audio, video_file, manager, cache, pool)
            cached_whisper = cache.load_json(video_file, "whisper")
            if cached_whisper is not None:
                whisper_segments = cached_whisper
                for index, segment in enumerate(whisper_segments):
                    segment["global_idx"] = index
            else:
                count = safe_instance_count(runtime.whisper_workers, whisper_model_name, device)
                if count != runtime.whisper_workers:
                    print(f"[Whisper] Capacity probe limited concurrency to {count} instance(s).")
                model = load_model(whisper_model_name, device)
                # Gemini calls run while this single full-track Whisper pass executes.
                whisper_segments = transcribe_full(audio, video_file, model, cache, WhisperGate(count))
        # OCR starts only after Whisper releases the accelerator, while outstanding Gemini calls continue.
        if run_ocr or ocr_only:
            ocr_cues, resolution = process_video_signs(video_file, manager, cache, pool, runtime.ocr_gpu)
            cues.extend(ocr_cues)
        used_indices = set()
        for job in gemini_jobs:  # Preserve chunk order even though calls execute concurrently.
            data, start, end = job.result()
            window = [s for s in whisper_segments if s.get("end", 0) >= start - 15 and s.get("start", 0) <= end + 15]
            chunk_cues, chunk_used, recovery = align_subtitles(data or [], window, start, end)
            cues.extend(chunk_cues)
            used_indices.update(chunk_used)
            for gap_start, gap_end, missing in recovery:
                print(f"[Alignment] Bounded {len(missing)} unmatched line(s) to {gap_start:.2f}-{gap_end:.2f}s")
        # Re-translate long unused Whisper regions. Calls are submitted together,
        # so independent recovery regions use the same Gemini worker pool.
        orphan_blocks, current = [], []
        for segment in whisper_segments:
            if segment.get("global_idx") not in used_indices and len(str(segment.get("text", "")).strip()) > 1:
                current.append(segment)
            elif current:
                orphan_blocks.append(current)
                current = []
        if current:
            orphan_blocks.append(current)
        recovery_jobs = []
        for index, block in enumerate(orphan_blocks, 1):
            start, end = float(block[0]["start"]), float(block[-1]["end"])
            if end - start > ORPHAN_GAP_THRESH_SEC:
                recovery_jobs.append((block, pool.submit(translate_recovery_slice, audio, video_file, start, end,
                                                        f"gemini_recovery_{round(start * 1000)}_{round(end * 1000)}",
                                                        manager, cache)))
        for block, job in recovery_jobs:
            data, start, end = job.result()
            recovered, recovered_used, _ = align_subtitles(data or [], block, start, end)
            cues.extend(recovered)
            used_indices.update(recovered_used)
    # Overlapping Gemini chunks can repeat the same utterance. Keep the best
    # first cue when normalized text and timing substantially overlap.
    deduplicated = []
    for cue in sorted(cues, key=lambda item: (item.start, item.layer)):
        duplicate = next((old for old in deduplicated if old.text == cue.text and
                          min(old.end, cue.end) - max(old.start, cue.start) > 0.25), None)
        if duplicate is None:
            deduplicated.append(cue)
    cues = deduplicated
    if runtime.strict_timing and not ocr_only:
        refine_contiguous_starts(cues, audio)
    if output_path.lower().endswith(".ass"):
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
            process_video(str(media), str(output), run_ocr, ocr_only, use_lite, runtime, model, device)
