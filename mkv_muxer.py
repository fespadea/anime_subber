import os
import sys
import argparse
import subprocess
import shutil

SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')

def check_ffmpeg():
    """Verify that FFmpeg is installed and accessible in the system path."""
    if not shutil.which("ffmpeg"):
        print("Error: 'ffmpeg' is not installed or not found in your system PATH.")
        print("This script requires FFmpeg to efficiently mux the video and subtitles.")
        print("Please install it (e.g., 'winget install ffmpeg' on Windows) and try again.")
        sys.exit(1)

def process_directory(target_dir):
    # Strip accidental quotes from drag-and-drop paths
    target_dir = target_dir.strip('"\'')
    
    if not os.path.isdir(target_dir):
        print(f"Error: The path '{target_dir}' is not a valid directory.")
        return

    output_dir = os.path.join(target_dir, "Muxed_Output")
    found_pairs = False

    print(f"Scanning directory: {target_dir}")

    for file in sorted(os.listdir(target_dir)):
        if file.lower().endswith(SUPPORTED_EXTS):
            base_name = os.path.splitext(file)[0]
            video_path = os.path.join(target_dir, file)
            srt_path = os.path.join(target_dir, base_name + ".srt")

            if os.path.exists(srt_path):
                if not found_pairs:
                    # Create the clean subfolder only when we actually find files to process
                    os.makedirs(output_dir, exist_ok=True)
                    found_pairs = True
                    print(f"Created output folder: {output_dir}\n")

                out_mkv = os.path.join(output_dir, base_name + ".mkv")

                if os.path.exists(out_mkv):
                    print(f"Skipping: {base_name} (MKV already exists in output folder)")
                    continue

                print(f"Muxing: {base_name}...")

                # Construct the FFmpeg command
                # -c copy: Copies video/audio streams natively without re-encoding (instant, lossless)
                # -c:s ass: Converts the SRT into Advanced SubStation Alpha format inside the MKV.
                #           This ensures {\an8} positioning tags and overlapping OCR subtitles 
                #           are natively respected by all modern video players.
                cmd = [
                    "ffmpeg",
                    "-y",                 # Overwrite output files without asking
                    "-i", video_path,     # Input 0: The video file
                    "-i", srt_path,       # Input 1: The SRT file
                    "-map", "0",          # Take all streams (video, audio, etc) from input 0
                    "-map", "1:0",        # Take the first subtitle stream from input 1
                    "-c", "copy",         # Do not re-encode video or audio
                    "-c:s", "ass",        # Convert SRT text stream to ASS format
                    "-metadata:s:s:0", "language=eng", 
                    "-metadata:s:s:0", "title=English (AI Translated)",
                    out_mkv
                ]

                try:
                    # Run the command and suppress the massive wall of FFmpeg text output
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
                    print(f"  -> Success!")
                except subprocess.CalledProcessError as e:
                    print(f"  -> Error muxing {base_name}: {e}")

    if not found_pairs:
        print("No matching Video + SRT pairs were found in the provided directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mux anime videos and SRT files into MKVs, preserving formatting.")
    parser.add_argument("path", help="Path to the directory containing your videos and SRT files.")
    args = parser.parse_args()
    
    check_ffmpeg()
    process_directory(args.path)
