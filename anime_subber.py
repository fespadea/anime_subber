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
from pydub import AudioSegment
from thefuzz import fuzz

from google import genai
from google.genai import types

# Suppress verbose EasyOCR warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
client = genai.Client()
SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')

CHUNK_LENGTH_MS = 2 * 60 * 1000  # 2 minutes
OVERLAP_MS = 30 * 1000           # 30 seconds overlap
STEP_MS = CHUNK_LENGTH_MS - OVERLAP_MS

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
    """Converts SRT timestamp (HH:MM:SS,mmm) to seconds."""
    h, m, s = time_str.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_srt(filepath):
    """Parses an existing SRT file into a list of subtitle dictionaries."""
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
            print(f"Failed to parse JSON completely. Error: {final_e}")
            return []

def get_numpad_position(bbox, width, height):
    """Maps an EasyOCR bounding box to a numpad position (1-9) for SRT {\\anX} tags."""
    tl, tr, br, bl = bbox
    center_x = (tl[0] + br[0]) / 2
    center_y = (tl[1] + br[1]) / 2
    
    col = 1 if center_x < width / 3 else (3 if center_x > 2 * width / 3 else 2)
    if center_y < height / 3:
        row = 3  # Top of screen
    elif center_y > 2 * height / 3:
        row = 1  # Bottom of screen
    else:
        row = 2  # Middle of screen
        
    # Numpad mapping grid
    numpad = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return numpad[row-1][col-1]

def align_audio_subs(gemini_data, whisper_segments, last_global_end_time):
    """Matches Gemini translations to the absolute Whisper timestamps."""
    srt_blocks = []
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
        
        max_search_ahead = min(last_match_idx + 20, len(whisper_segments))
        
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
            
            # Since Whisper was run on the full file, these timestamps are absolute. No offset needed.
            start_time = matched_start_seg['start']
            end_time = matched_end_seg['end']
            
            # Prevent overlap duplicates from the sliding chunk window
            if start_time < last_global_end_time - 0.5:
                start_time = max(start_time, last_global_end_time + 0.001)
                if start_time >= end_time:
                    continue 
                    
            srt_blocks.append({
                'start': start_time,
                'end': end_time,
                'text': en_text,
                'pos': None # Audio doesn't get numpad positioning
            })
            print(f"[{format_srt_time(start_time).split(',')[0]}] {en_text}")
            
            last_global_end_time = end_time 
            
    return srt_blocks, last_global_end_time

def process_video_signs(video_file):
    """Scans the video for Japanese text, groups consecutive sightings, and translates via Gemini."""
    import easyocr
    print("\n--- Starting Vision Pass for On-Screen Text ---")
    print("Loading EasyOCR (This requires significant VRAM)...")
    reader = easyocr.Reader(['ja'], gpu=True)
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    frame_interval = int(fps) # Scan 1 frame per second
    current_signs = []
    final_signs = []
    
    frame_count = 0
    print("Scanning video frames for Japanese text...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        if frame_count % frame_interval == 0:
            current_time_sec = frame_count / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_texts = reader.readtext(gray, detail=1)
            
            for bbox, text, conf in detected_texts:
                if conf < 0.6: continue
                norm_t = normalize_text(text)
                if len(norm_t) < 2: continue # Ignore isolated single characters
                
                # Check if we are already tracking this sign
                found = False
                for active in current_signs:
                    if fuzz.ratio(norm_t, active['norm_text']) > 80:
                        active['end'] = current_time_sec + 1.0
                        found = True
                        break
                
                if not found:
                    current_signs.append({
                        'start': current_time_sec,
                        'end': current_time_sec + 1.0,
                        'ja_text': text,
                        'norm_text': norm_t,
                        'pos': get_numpad_position(bbox, width, height)
                    })
            
            # Move expired signs to the final list
            for active in current_signs[:]:
                if current_time_sec > active['end'] + 1.5:
                    final_signs.append(active)
                    current_signs.remove(active)
                    
        frame_count += 1
        
    cap.release()
    final_signs.extend(current_signs)
    
    ocr_srt_blocks = []
    if final_signs:
        print(f"\nFound {len(final_signs)} on-screen signs. Batch translating via Gemini...")
        
        # Batch to prevent token bloat
        batch_size = 50
        for i in range(0, len(final_signs), batch_size):
            batch = final_signs[i:i+batch_size]
            prompt = (
                "You are an expert anime translator. Translate the following list of Japanese on-screen text to English. "
                "Keep translations concise. Return ONLY a valid JSON array of strings in the exact same order as the input.\n"
            )
            prompt += json.dumps([s['ja_text'] for s in batch], ensure_ascii=False)
            
            generation_config = types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
            
            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=[prompt],
                    config=generation_config
                )
                
                translations = parse_llm_json(response.text)
                
                for sign, en_text in zip(batch, translations):
                    if en_text:
                        ocr_srt_blocks.append({
                            'start': sign['start'],
                            'end': sign['end'],
                            'text': f"[{en_text}]", # Brackets clearly denote on-screen text
                            'pos': sign['pos']
                        })
            except Exception as e:
                print(f"Error translating signs batch: {e}")
                
    return ocr_srt_blocks

def process_anime_video(video_file, output_srt, run_ocr=False, ocr_only=False):
    print(f"\nLoading {video_file}...")

    all_subtitles = []
    last_global_end_time = -1.0 

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

        # 1. UPFRONT WHISPER PASS ON FULL AUDIO
        print("\n--- Starting Audio Pass ---")
        whisper_cache_file = os.path.splitext(video_file)[0] + ".whisper.json"
        
        if os.path.exists(whisper_cache_file):
            print(f"Loading cached Whisper transcription from {whisper_cache_file}...")
            with open(whisper_cache_file, "r", encoding="utf-8") as f:
                global_whisper_segments = json.load(f)
        else:
            print("Loading local Whisper model (large) for perfect timestamping...")
            whisper_model = whisper.load_model("large", device="cuda")
            
            temp_full_audio = "temp_full_audio.wav"
            audio.export(temp_full_audio, format="wav")
            
            print("Transcribing full audio track (This takes a moment but saves time later)...")
            whisper_result = whisper_model.transcribe(temp_full_audio, language="ja")
            global_whisper_segments = whisper_result['segments']
            os.remove(temp_full_audio)
            
            print(f"Caching Whisper transcription to {whisper_cache_file}...")
            with open(whisper_cache_file, "w", encoding="utf-8") as f:
                json.dump(global_whisper_segments, f, ensure_ascii=False, indent=2)

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

        print("Beginning chunked Gemini translation...")
        
        for start_ms in range(0, len(audio), STEP_MS):
            end_ms = min(start_ms + CHUNK_LENGTH_MS, len(audio))
            chunk = audio[start_ms:end_ms]
            
            time_offset_seconds = start_ms / 1000.0
            
            chunk_num = (start_ms // STEP_MS) + 1
            total_chunks = (len(audio) // STEP_MS) + 1
            
            print(f"\nProcessing Audio Chunk {chunk_num}/{total_chunks}...")
            
            gemini_data = None
            gemini_cache_file = os.path.splitext(video_file)[0] + f".gemini_chunk_{chunk_num}.json"
            
            if os.path.exists(gemini_cache_file):
                print(f"Loading cached Gemini translation for chunk {chunk_num}...")
                try:
                    with open(gemini_cache_file, "r", encoding="utf-8") as f:
                        gemini_data = json.load(f)
                except Exception as e:
                    print(f"Error loading cache for chunk {chunk_num}: {e}. Re-running translation...")
                    gemini_data = None
            
            if not gemini_data:
                temp_audio_path = f"temp_chunk_{chunk_num}.wav"
                chunk.export(temp_audio_path, format="wav")
                
                try:
                    wav_data = open(temp_audio_path, "rb").read()
                    response = client.models.generate_content(
                        model="gemini-flash-latest",
                        contents=[
                            types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                            "Transcribe and translate this audio into the requested JSON format."
                        ],
                        config=generation_config
                    )

                    raw_text = response.text.strip()
                    try:
                        gemini_data = json.loads(raw_text)
                    except json.JSONDecodeError as e:
                        if "Extra data" in e.msg:
                            clean_text = raw_text[:e.pos].strip()
                            gemini_data = json.loads(clean_text)
                        else:
                            raise
                            
                    print(f"Caching Gemini translation for chunk {chunk_num}...")
                    with open(gemini_cache_file, "w", encoding="utf-8") as f:
                        json.dump(gemini_data, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"Error processing audio chunk {chunk_num}: {e}")
                    
                finally:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            
            if gemini_data:
                # Filter the global whisper segments to only those inside this chunk's timeframe (+15s buffer)
                chunk_w_segs = [seg for seg in global_whisper_segments if 
                                seg['start'] >= (time_offset_seconds - 15) and 
                                seg['end'] <= (end_ms / 1000.0 + 15)]
                
                chunk_srt_lines, last_global_end_time = align_audio_subs(
                    gemini_data, chunk_w_segs, last_global_end_time
                )
                all_subtitles.extend(chunk_srt_lines)
                    
            if end_ms >= len(audio):
                break

    # 2. OPTIONAL VISION PASS
    if run_ocr or ocr_only:
        ocr_subtitles = process_video_signs(video_file)
        all_subtitles.extend(ocr_subtitles)

    # 3. SORT AND FORMAT FINAL SRT
    if all_subtitles:
        # Sort chronologically so audio and OCR text are perfectly interwoven
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

def process_target_path(target_path, run_ocr=False, ocr_only=False):
    if os.path.isfile(target_path):
        if target_path.lower().endswith(SUPPORTED_EXTS):
            output_srt = os.path.splitext(target_path)[0] + ".srt"
            if not os.path.exists(output_srt) or ocr_only:
                process_anime_video(target_path, output_srt, run_ocr, ocr_only)
            else:
                print(f"Skipping: {target_path} (SRT already exists)")
    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTS):
                    video_path = os.path.join(root, file)
                    output_srt = os.path.splitext(video_path)[0] + ".srt"
                    if not os.path.exists(output_srt) or ocr_only:
                        process_anime_video(video_path, output_srt, run_ocr, ocr_only)
                    else:
                        print(f"Skipping: {file} (SRT already exists)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anime subtitles via Forced Alignment.")
    parser.add_argument("path", help="Path to video file or directory.")
    parser.add_argument("--ocr", action="store_true", help="Enable vision pass to detect and translate on-screen Japanese text.")
    parser.add_argument("--ocr_only", action="store_true", help="Only run the vision pass. Appends to existing SRT and creates backup.")
    args = parser.parse_args()
    process_target_path(args.path, args.ocr, args.ocr_only)
