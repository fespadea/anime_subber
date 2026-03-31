import os
import io
import json
import argparse
import datetime
import whisper
import re
import cv2
import numpy as np
import shutil
import concurrent.futures
from pydub import AudioSegment
from thefuzz import fuzz

from google import genai
from google.genai import types

# Suppress verbose EasyOCR warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
# NOTE: You must set the GEMINI_API_KEY environment variable before running this script.
# Windows (CMD): set GEMINI_API_KEY="your_api_key_here"
# Windows (PowerShell): $env:GEMINI_API_KEY="your_api_key_here"
# Mac/Linux: export GEMINI_API_KEY="your_api_key_here"

client = genai.Client()
SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav')

CHUNK_LENGTH_MS = 4 * 60 * 1000  # 4 minutes
OVERLAP_MS = 30 * 1000           # 30 seconds overlap
STEP_MS = CHUNK_LENGTH_MS - OVERLAP_MS
ORPHAN_GAP_THRESH_SEC = 20.0     # Threshold in seconds to trigger a re-translation of missing dialogue

class GeminiManager:
    """Handles API calls, quota fallbacks, and the kill-switch if limits are exceeded."""
    def __init__(self, use_lite=False):
        self.api_exhausted = False
        self.models = ["gemini-flash-latest", "gemini-flash-lite-latest"]
        if use_lite:
            self.models = ["gemini-flash-lite-latest"]
            
    def generate(self, contents, config, prefer_lite=False):
        if self.api_exhausted or not self.models:
            return None
        
        models_to_try = self.models
        if prefer_lite:
            # Reorder list to try lite models first for simpler tasks like OCR
            lite_models = [m for m in self.models if "lite" in m]
            standard_models = [m for m in self.models if "lite" not in m]
            models_to_try = lite_models + standard_models
        
        # Iterate over a copy using list() so we can safely remove items from self.models
        for model_name in list(models_to_try):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    print(f"  [Gemini] Model {model_name} exhausted or rate-limited. Removing from rotation.")
                    if model_name in self.models:
                        self.models.remove(model_name)
                    continue
                else:
                    print(f"  [Gemini] Error with {model_name}: {e}")
                    continue
        
        if not self.models:
            print("\n[!] All available Gemini models exhausted their API quota. Suspending API calls.")
            self.api_exhausted = True
            
        return None

def format_srt_time(seconds):
    delta = datetime.timedelta(seconds=seconds)
    time_str = str(delta)
    if '.' in time_str:
        time_str, ms_str = time_str.split('.')
        ms_str = ms_str[:3].ljust(3, '0')
    else:
        ms_str = "000"
    
    parts = time_str.split(':')
    if len(parts) == 2:
        time_str = f"00:{parts[0]:0>2}:{parts[1]:0>2}"
    else:
        time_str = f"{parts[0]:0>2}:{parts[1]:0>2}:{parts[2]:0>2}"
        
    return f"{time_str},{ms_str}"

def srt_time_to_seconds(time_str):
    h, m, s = time_str.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_srt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip().split('\n\n')
    subs = []
    for block in content:
        lines = block.split('\n')
        if len(lines) >= 3:
            times = lines[1].split(' --> ')
            if len(times) == 2:
                start = srt_time_to_seconds(times[0].strip())
                end = srt_time_to_seconds(times[1].strip())
                text = '\n'.join(lines[2:])
                subs.append({'start': start, 'end': end, 'text': text, 'pos': None})
    return subs

def normalize_text(text):
    return re.sub(r'[\s\u3000、。！？「」『』（）,\.\?!♪~～]', '', text)

def parse_llm_json(text):
    text = text.strip()
    md_fence = "`" * 3
    if text.startswith(f'{md_fence}json'): text = text[7:]
    elif text.startswith(md_fence): text = text[3:]
    if text.endswith(md_fence): text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "Extra data" in e.msg:
            try:
                return json.loads(text[:e.pos].strip())
            except Exception:
                pass
        text_cleaned = re.sub(r'\]\s*\]$', ']', text)
        try:
            return json.loads(text_cleaned)
        except Exception as final_e:
            return []

def get_numpad_position(bbox, width, height):
    tl, tr, br, bl = bbox
    center_x = (tl[0] + br[0]) / 2
    center_y = (tl[1] + br[1]) / 2
    
    col = 1 if center_x < width / 3 else (3 if center_x > 2 * width / 3 else 2)
    if center_y < height / 3:
        row = 3 
    elif center_y > 2 * height / 3:
        row = 1 
    else:
        row = 2 
        
    numpad = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return numpad[row-1][col-1]

def refine_early_timestamps(start_time, end_time, full_audio, prev_end_time):
    """Trims leading noise/silence if a subtitle starts too early or immediately after the previous one."""
    if start_time == 0 or start_time == prev_end_time:
        try:
            check_duration = min(1.0, end_time - start_time)
            if check_duration <= 0.1: return start_time
            
            segment = full_audio[int(start_time * 1000):int((start_time + check_duration) * 1000)]
            peak = segment.max_dBFS
            if peak == float('-inf'): return start_time
            
            # Scan in 50ms windows to find when the audio gets within 15dB of the peak
            window_ms = 50
            for i in range(0, len(segment), window_ms):
                window = segment[i:i+window_ms]
                if window.max_dBFS > peak - 15:
                    return start_time + (i / 1000.0)
        except Exception:
            pass 
    return start_time

def align_audio_subs(gemini_data, whisper_segments, last_global_end_time, strict_timing=False, full_audio=None):
    srt_blocks = []
    used_whisper_indices = set()
    last_match_idx = 0
    
    for g_line in gemini_data:
        ja_text = g_line.get("ja", "")
        en_text = g_line.get("en", "")
        
        if not ja_text or not en_text or "[NO SPEECH]" in en_text:
            continue
            
        norm_ja = normalize_text(ja_text)
        if not norm_ja: continue
            
        best_match_score = 0
        best_start_idx = -1
        best_end_idx = -1
        
        # Increased to 100. Whisper often generates dozens of tiny fragmented 
        # noise/music segments. We need a large enough search window to jump over them.
        max_search_ahead = min(last_match_idx + 100, len(whisper_segments))
        
        for i in range(last_match_idx, max_search_ahead):
            combined_text = ""
            for j in range(i, min(i + 6, len(whisper_segments))):
                combined_text += whisper_segments[j]['text']
                norm_combined = normalize_text(combined_text)
                if not norm_combined: continue
                
                base_score = fuzz.partial_ratio(norm_ja, norm_combined)
                size_ratio = min(len(norm_ja), len(norm_combined)) / max(len(norm_ja), len(norm_combined))
                final_score = base_score * (size_ratio ** 0.5) 
                
                if final_score > best_match_score:
                    best_match_score = final_score
                    best_start_idx = i
                    best_end_idx = j
                    
        if best_start_idx != -1 and best_match_score > 55:
            matched_start_seg = whisper_segments[best_start_idx]
            matched_end_seg = whisper_segments[best_end_idx]
            last_match_idx = best_end_idx + 1 
            
            start_time = matched_start_seg['start']
            end_time = matched_end_seg['end']
            
            if strict_timing and full_audio is not None:
                start_time = refine_early_timestamps(start_time, end_time, full_audio, last_global_end_time)
            
            if start_time < last_global_end_time - 0.5:
                start_time = max(start_time, last_global_end_time + 0.001)
                if start_time >= end_time:
                    continue 
                    
            srt_blocks.append({
                'start': start_time,
                'end': end_time,
                'text': en_text,
                'pos': None 
            })
            
            # Track which global indices were used
            used_whisper_indices.update(whisper_segments[idx]['global_idx'] for idx in range(best_start_idx, best_end_idx + 1))
            last_global_end_time = end_time 
            
    return srt_blocks, last_global_end_time, used_whisper_indices

def run_whisper_pass(audio, video_file):
    whisper_cache_file = os.path.splitext(video_file)[0] + ".whisper.json"
    
    if os.path.exists(whisper_cache_file):
        print(f"[Whisper] Loading cached transcription from {whisper_cache_file}...")
        with open(whisper_cache_file, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        print("[Whisper] Loading local model (large) for perfect timestamping...")
        whisper_model = whisper.load_model("large", device="cuda")
        
        temp_full_audio = "temp_full_audio.wav"
        audio.export(temp_full_audio, format="wav")
        
        print("[Whisper] Transcribing full audio track...")
        whisper_result = whisper_model.transcribe(temp_full_audio, language="ja")
        segments = whisper_result['segments']
        os.remove(temp_full_audio)
        
        print(f"[Whisper] Caching transcription to {whisper_cache_file}...")
        with open(whisper_cache_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

    # Assign a global ID so we can track orphans later
    for i, seg in enumerate(segments):
        seg['global_idx'] = i
        
    return segments

def run_gemini_pass(audio, video_file, gemini_manager, config):
    gemini_chunks_data = []
    
    for start_ms in range(0, len(audio), STEP_MS):
        end_ms = min(start_ms + CHUNK_LENGTH_MS, len(audio))
        chunk = audio[start_ms:end_ms]
        time_offset_seconds = start_ms / 1000.0
        
        chunk_num = (start_ms // STEP_MS) + 1
        total_chunks = (len(audio) // STEP_MS) + 1
        
        gemini_cache_file = os.path.splitext(video_file)[0] + f".gemini_chunk_{chunk_num}.json"
        gemini_data = None
        
        if os.path.exists(gemini_cache_file):
            print(f"[Gemini] Loading cached translation for chunk {chunk_num}/{total_chunks}...")
            try:
                with open(gemini_cache_file, "r", encoding="utf-8") as f:
                    gemini_data = json.load(f)
            except Exception as e:
                print(f"[Gemini] Error loading cache for chunk {chunk_num}: {e}. Re-running...")
        
        if not gemini_data and not gemini_manager.api_exhausted:
            print(f"[Gemini] Processing Audio Chunk {chunk_num}/{total_chunks}...")
            temp_audio_path = f"temp_chunk_{chunk_num}.wav"
            chunk.export(temp_audio_path, format="wav")
            
            try:
                wav_data = open(temp_audio_path, "rb").read()
                response = gemini_manager.generate(
                    contents=[
                        types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                        "Transcribe and translate this audio into the requested JSON format."
                    ],
                    config=config
                )

                if response:
                    raw_text = response.text.strip()
                    try:
                        gemini_data = json.loads(raw_text)
                    except json.JSONDecodeError as e:
                        if "Extra data" in e.msg:
                            clean_text = raw_text[:e.pos].strip()
                            gemini_data = json.loads(clean_text)
                        else:
                            gemini_data = parse_llm_json(raw_text)
                            
                    if gemini_data:
                        print(f"[Gemini] Caching translation for chunk {chunk_num}...")
                        with open(gemini_cache_file, "w", encoding="utf-8") as f:
                            json.dump(gemini_data, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"[Gemini] Error processing audio chunk {chunk_num}: {e}")
                
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        gemini_chunks_data.append((gemini_data, time_offset_seconds, end_ms / 1000.0))
        if end_ms >= len(audio):
            break
            
    return gemini_chunks_data

def recheck_missing_dialogue(audio, whisper_segments, used_indices, gemini_manager, config, strict_timing, video_file):
    print("\n--- Scanning for Missing Dialogue ---")
    new_subtitles = []
    orphan_blocks = []
    current_block = []
    
    for seg in whisper_segments:
        if seg['global_idx'] not in used_indices:
            text = normalize_text(seg['text'])
            if len(text) > 1:  # Contains actual content, not just music symbols
                current_block.append(seg)
        else:
            if current_block:
                orphan_blocks.append(current_block)
                current_block = []
                
    if current_block:
        orphan_blocks.append(current_block)
        
    chunk_sub_counters = {}  # Tracks the Y counter for each X chunk
        
    for block in orphan_blocks:
        start_seg = block[0]
        end_seg = block[-1]
        duration = end_seg['end'] - start_seg['start']
        
        # If the missing gap is significant (> ORPHAN_GAP_THRESH_SEC)
        if duration > ORPHAN_GAP_THRESH_SEC:
            start_ms_calc = int(start_seg['start'] * 1000)
            chunk_x = (start_ms_calc // STEP_MS) + 1
            chunk_y = chunk_sub_counters.get(chunk_x, 1)
            chunk_sub_counters[chunk_x] = chunk_y + 1
            
            gemini_cache_file = os.path.splitext(video_file)[0] + f".gemini_chunk_{chunk_x}.{chunk_y}.json"
            gemini_data = None
            
            time_start_str = format_srt_time(start_seg['start']).split(',')[0]
            print(f"Targeting missing dialogue at {time_start_str} (Chunk {chunk_x}.{chunk_y})...")
            
            if os.path.exists(gemini_cache_file):
                print(f"  [Gemini] Loading cached missing dialogue translation from {gemini_cache_file}...")
                try:
                    with open(gemini_cache_file, "r", encoding="utf-8") as f:
                        gemini_data = json.load(f)
                except Exception as e:
                    print(f"  [Gemini] Error loading cache for chunk {chunk_x}.{chunk_y}: {e}. Re-running...")
            
            if not gemini_data and not gemini_manager.api_exhausted:
                start_ms = max(0, int(start_seg['start'] * 1000) - 500)
                end_ms = min(len(audio), int(end_seg['end'] * 1000) + 500)
                
                chunk = audio[start_ms:end_ms]
                wav_io = io.BytesIO()
                chunk.export(wav_io, format="wav")
                
                response = gemini_manager.generate(
                    contents=[
                        types.Part.from_bytes(data=wav_io.getvalue(), mime_type="audio/wav"),
                        "Transcribe and translate this missed audio segment into the requested JSON format."
                    ],
                    config=config
                )
                
                if response:
                    raw_text = response.text.strip()
                    try:
                        gemini_data = json.loads(raw_text)
                    except json.JSONDecodeError as e:
                        if "Extra data" in e.msg:
                            clean_text = raw_text[:e.pos].strip()
                            gemini_data = json.loads(clean_text)
                        else:
                            gemini_data = parse_llm_json(raw_text)
                            
                    if gemini_data:
                        print(f"  [Gemini] Caching missing dialogue translation to {gemini_cache_file}...")
                        with open(gemini_cache_file, "w", encoding="utf-8") as f:
                            json.dump(gemini_data, f, ensure_ascii=False, indent=2)
            
            if gemini_data:
                subs, _, _ = align_audio_subs(gemini_data, block, -1.0, strict_timing, audio)
                new_subtitles.extend(subs)
                    
    return new_subtitles

def process_video_signs(video_file, gemini_manager):
    import easyocr
    print("\n--- Starting Vision Pass for On-Screen Text ---")
    print("Loading EasyOCR (This requires significant VRAM)...")
    reader = easyocr.Reader(['ja'], gpu=True)
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    frame_interval = int(fps) 
    scan_interval_sec = frame_interval / fps
    current_signs = []
    final_signs = []
    
    frame_count = 0
    print("Scanning video frames for Japanese text (approx 1 frame/sec)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if frame_count % frame_interval == 0:
            current_time_sec = frame_count / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_texts = reader.readtext(gray, detail=1)
            
            seen_this_frame = []
            need_restore_position = False
            
            for bbox, text, conf in detected_texts:
                if conf < 0.6: continue
                norm_t = normalize_text(text)
                if len(norm_t) < 2: continue 
                
                found = False
                for active in current_signs:
                    if fuzz.ratio(norm_t, active['norm_text']) > 80:
                        active['last_seen'] = current_time_sec
                        active['bbox'] = bbox
                        seen_this_frame.append(active)
                        found = True
                        break
                
                if not found:
                    # Binary search backwards to find EXACT start time
                    left_start = max(0.0, current_time_sec - scan_interval_sec - 0.1)
                    right_start = current_time_sec
                    
                    for _ in range(4): # 4 steps = ~0.06s precision
                        mid = (left_start + right_start) / 2.0
                        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
                        ret_mid, f_mid = cap.read()
                        if not ret_mid:
                            right_start = mid
                            continue
                            
                        g_mid = cv2.cvtColor(f_mid, cv2.COLOR_BGR2GRAY)
                        tl, tr, br, bl = bbox
                        x_min = max(0, int(min(tl[0], bl[0])) - 20)
                        x_max = min(int(width), int(max(tr[0], br[0])) + 20)
                        y_min = max(0, int(min(tl[1], tr[1])) - 20)
                        y_max = min(int(height), int(max(bl[1], br[1])) + 20)
                        
                        if x_max <= x_min or y_max <= y_min:
                            right_start = mid
                            continue
                            
                        crop = g_mid[y_min:y_max, x_min:x_max]
                        if crop.size == 0:
                            right_start = mid
                            continue
                            
                        det = reader.readtext(crop, detail=0)
                        found_in_mid = False
                        for dt in det:
                            if fuzz.ratio(normalize_text(dt), norm_t) > 80:
                                found_in_mid = True
                                break
                                
                        if found_in_mid:
                            right_start = mid 
                        else:
                            left_start = mid 
                            
                    new_sign = {
                        'start': right_start,
                        'last_seen': current_time_sec,
                        'ja_text': text,
                        'norm_text': norm_t,
                        'bbox': bbox,
                        'pos': get_numpad_position(bbox, width, height)
                    }
                    current_signs.append(new_sign)
                    seen_this_frame.append(new_sign)
                    need_restore_position = True
            
            for active in current_signs[:]:
                if active not in seen_this_frame:
                    # Binary search forwards to find EXACT end time
                    left_end = active['last_seen']
                    right_end = current_time_sec
                    
                    for _ in range(4):
                        mid = (left_end + right_end) / 2.0
                        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
                        ret_mid, f_mid = cap.read()
                        if not ret_mid:
                            right_end = mid
                            continue
                            
                        g_mid = cv2.cvtColor(f_mid, cv2.COLOR_BGR2GRAY)
                        tl, tr, br, bl = active['bbox']
                        x_min = max(0, int(min(tl[0], bl[0])) - 20)
                        x_max = min(int(width), int(max(tr[0], br[0])) + 20)
                        y_min = max(0, int(min(tl[1], tr[1])) - 20)
                        y_max = min(int(height), int(max(bl[1], br[1])) + 20)
                        
                        if x_max <= x_min or y_max <= y_min:
                            right_end = mid
                            continue
                            
                        crop = g_mid[y_min:y_max, x_min:x_max]
                        if crop.size == 0:
                            right_end = mid
                            continue
                            
                        det = reader.readtext(crop, detail=0)
                        found_in_mid = False
                        for dt in det:
                            if fuzz.ratio(normalize_text(dt), active['norm_text']) > 80:
                                found_in_mid = True
                                break
                                
                        if found_in_mid:
                            left_end = mid
                        else:
                            right_end = mid
                            
                    active['end'] = left_end
                    final_signs.append(active)
                    current_signs.remove(active)
                    need_restore_position = True
                    
            if need_restore_position:
                # Restore the sequential frame reading back to where we paused it
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 1)
                
        frame_count += 1
        
    cap.release()
    
    # Sweep up any signs that were still on screen when the video ended
    for active in current_signs:
        active['end'] = active['last_seen']
        final_signs.append(active)

    # --- COMBINE AND SORT OVERLAPPING SIGNS ---
    print("Combining overlapping on-screen text...")
    combined_signs = []
    
    for sign in sorted(final_signs, key=lambda x: x['start']):
        overlap_found = False
        for comb in combined_signs:
            if comb['pos'] == sign['pos']:
                # If they overlap in time
                start_max = max(comb['start'], sign['start'])
                end_min = min(comb['end'], sign['end'])
                if end_min > start_max:
                    comb['start'] = min(comb['start'], sign['start'])
                    comb['end'] = max(comb['end'], sign['end'])
                    comb['lines'].append(sign)
                    overlap_found = True
                    break
                    
        if not overlap_found:
            combined_signs.append({
                'start': sign['start'],
                'end': sign['end'],
                'pos': sign['pos'],
                'lines': [sign]
            })
            
    # Sort them vertically based on their bounding box Y-coordinates
    for comb in combined_signs:
        comb['lines'].sort(key=lambda x: (x['bbox'][0][1] + x['bbox'][2][1]) / 2)
        comb['ja_text'] = "\n".join([x['ja_text'] for x in comb['lines']])
    
    ocr_srt_blocks = []
    if combined_signs:
        print(f"\nFound {len(combined_signs)} grouped on-screen signs. Batch translating via Gemini...")
        batch_size = 50
        for i in range(0, len(combined_signs), batch_size):
            batch = combined_signs[i:i+batch_size]
            prompt = (
                "You are an expert anime translator. Translate the following list of Japanese on-screen text to English. "
                "Keep translations concise. Return ONLY a valid JSON array of strings in the exact same order as the input.\n"
                "Note: Some items may contain multiple lines of text separated by newlines. Translate them as a single combined block.\n"
            )
            prompt += json.dumps([s['ja_text'] for s in batch], ensure_ascii=False)
            
            generation_config = types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
            
            try:
                response = gemini_manager.generate(
                    contents=[prompt],
                    config=generation_config,
                    prefer_lite=True
                )
                
                if response:
                    translations = parse_llm_json(response.text)
                    for sign, en_text in zip(batch, translations):
                        if en_text:
                            # Wrap each line in brackets so the viewer clearly knows it's an on-screen sign 
                            formatted_text = "\n".join([f"[{line}]" for line in en_text.split('\n')]) if '\n' in en_text else f"[{en_text}]"
                            
                            ocr_srt_blocks.append({
                                'start': sign['start'],
                                'end': sign['end'],
                                'text': formatted_text, 
                                'pos': sign['pos']
                            })
            except Exception as e:
                print(f"Error translating signs batch: {e}")
                
    return ocr_srt_blocks

def process_anime_video(video_file, output_srt, run_ocr=False, ocr_only=False, strict_timing=False, use_lite=False):
    print(f"\nLoading {video_file}...")
    all_subtitles = []
    gemini_manager = GeminiManager(use_lite=use_lite)

    if ocr_only:
        print("\n--- OCR Only Mode ---")
        if os.path.exists(output_srt):
            backup_file = output_srt + ".bak"
            shutil.copy2(output_srt, backup_file)
            print(f"Backed up existing SRT to {backup_file}")
            all_subtitles = parse_srt(output_srt)
            print(f"Loaded {len(all_subtitles)} existing subtitles.")
    else:
        try:
            audio = AudioSegment.from_file(video_file)
        except Exception as e:
            print(f"Failed to extract audio: {e}")
            return
        
        system_instruction = (
            "You are an expert anime translator. You will receive an audio clip. "
            "Transcribe the spoken Japanese, and translate it into natural English. "
            "Ignore background music and sound effects. "
            "You MUST respond ONLY with a valid JSON array of objects. "
            "Each object must have two keys: 'ja' (the Japanese transcription) and 'en' (the English translation). "
            "Example: [{\"ja\": \"Nani?\", \"en\": \"What?\"}]"
        )
        
        generation_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json"
        )

        print("\n--- Starting Parallel Audio Processing ---")
        # Run Whisper and Gemini tasks in parallel to save time
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_whisper = executor.submit(run_whisper_pass, audio, video_file)
            future_gemini = executor.submit(run_gemini_pass, audio, video_file, gemini_manager, generation_config)
            
            global_whisper_segments = future_whisper.result()
            gemini_chunks_data = future_gemini.result()

        print("\n--- Starting Alignment Pass ---")
        last_global_end_time = -1.0 
        used_whisper_indices = set()
        
        # Sequentially align the translated chunks against the Whisper blueprint
        for gemini_data, start_sec, end_sec in gemini_chunks_data:
            if not gemini_data: continue
            
            chunk_w_segs = [seg for seg in global_whisper_segments if 
                            seg['start'] >= (start_sec - 15) and 
                            seg['end'] <= (end_sec + 15)]
            
            chunk_srt_lines, last_global_end_time, chunk_used_idx = align_audio_subs(
                gemini_data, chunk_w_segs, last_global_end_time, strict_timing, audio
            )
            all_subtitles.extend(chunk_srt_lines)
            used_whisper_indices.update(chunk_used_idx)
            
        # Re-check orphaned Whisper segments
        missing_subs = recheck_missing_dialogue(audio, global_whisper_segments, used_whisper_indices, gemini_manager, generation_config, strict_timing, video_file)
        if missing_subs:
            all_subtitles.extend(missing_subs)

    if run_ocr or ocr_only:
        ocr_subtitles = process_video_signs(video_file, gemini_manager)
        all_subtitles.extend(ocr_subtitles)

    if all_subtitles:
        all_subtitles.sort(key=lambda x: x['start'])
        
        final_srt_lines = []
        for i, sub in enumerate(all_subtitles, 1):
            start_str = format_srt_time(sub['start'])
            end_str = format_srt_time(sub['end'])
            
            text = sub['text']
            if sub.get('pos'):
                text = f"{{\\an{sub['pos']}}}{text}"
                
            final_srt_lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")

        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(final_srt_lines))
        print(f"\nSuccess! Saved completely integrated subtitles to {output_srt}")
    else:
        print("\nFailed to generate any subtitles.")

def process_target_path(target_path, run_ocr=False, ocr_only=False, strict_timing=False, use_lite=False, force_update=False):
    if os.path.isfile(target_path):
        if target_path.lower().endswith(SUPPORTED_EXTS):
            output_srt = os.path.splitext(target_path)[0] + ".srt"
            if not os.path.exists(output_srt) or ocr_only or force_update:
                process_anime_video(target_path, output_srt, run_ocr, ocr_only, strict_timing, use_lite)
            else:
                print(f"Skipping: {target_path} (SRT already exists)")
    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTS):
                    video_path = os.path.join(root, file)
                    output_srt = os.path.splitext(video_path)[0] + ".srt"
                    if not os.path.exists(output_srt) or ocr_only or force_update:
                        process_anime_video(video_path, output_srt, run_ocr, ocr_only, strict_timing, use_lite)
                    else:
                        print(f"Skipping: {file} (SRT already exists)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anime subtitles via Forced Alignment.")
    parser.add_argument("path", help="Path to video file or directory.")
    parser.add_argument("--ocr", action="store_true", help="Enable vision pass to detect and translate on-screen Japanese text.")
    parser.add_argument("--ocr_only", action="store_true", help="Only run the vision pass. Appends to existing SRT and creates backup.")
    parser.add_argument("--strict_timing", action="store_true", help="Analyze audio volume to trim silent intros from early subtitles.")
    parser.add_argument("--lite", action="store_true", help="Force the use of the cheaper gemini-flash-lite-latest model.")
    parser.add_argument("--force_update", action="store_true", help="Regenerate the SRT file even if it already exists, using cached API outputs where available.")
    args = parser.parse_args()
    process_target_path(args.path, args.ocr, args.ocr_only, args.strict_timing, args.lite, args.force_update)
