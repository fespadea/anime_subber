import os
import sys
import argparse
import subprocess
import shutil

SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
SUBTITLE_FORMAT_PRIORITY = ('.ass', '.ssa', '.srt', '.vtt')

def check_ffmpeg():
    """Verify that FFmpeg is installed and accessible in the system path."""
    if not shutil.which("ffmpeg"):
        print("Error: 'ffmpeg' is not installed or not found in your system PATH.")
        print("This script requires FFmpeg to efficiently mux the video and subtitles.")
        print("Please install it (e.g., 'winget install ffmpeg' on Windows) and try again.")
        sys.exit(1)

def find_subtitle(target_dir, base_name, subtitle_format="all"):
    """Return the highest-priority matching subtitle path, if one exists."""
    if subtitle_format == "all":
        extensions = SUBTITLE_FORMAT_PRIORITY
    else:
        extension = subtitle_format.lower().lstrip('.')
        extensions = (f".{extension}",)
    for extension in extensions:
        candidate = os.path.join(target_dir, base_name + extension)
        if os.path.isfile(candidate):
            return candidate
    return None


def process_directory(target_dir, subtitle_format="all"):
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
            subtitle_path = find_subtitle(target_dir, base_name, subtitle_format)

            if subtitle_path:
                if not found_pairs:
                    # Create the clean subfolder only when we actually find files to process
                    os.makedirs(output_dir, exist_ok=True)
                    found_pairs = True
                    print(f"Created output folder: {output_dir}\n")

                out_mkv = os.path.join(output_dir, base_name + ".mkv")

                if os.path.exists(out_mkv):
                    print(f"Skipping: {base_name} (MKV already exists in output folder)")
                    continue

                subtitle_ext = os.path.splitext(subtitle_path)[1].lstrip('.').upper()
                print(f"Muxing: {base_name} using {subtitle_ext} subtitles...")

                # Construct the FFmpeg command
                # -c copy: Copies video/audio streams natively without re-encoding (instant, lossless)
                # -c:s ass: Normalizes supported text subtitle inputs to ASS inside the MKV.
                cmd = [
                    "ffmpeg",
                    "-y",                 # Overwrite output files without asking
                    "-i", video_path,     # Input 0: The video file
                    "-i", subtitle_path,  # Input 1: The selected subtitle file
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
        requested = "supported subtitle" if subtitle_format == "all" else subtitle_format.upper()
        print(f"No matching Video + {requested} pairs were found in the provided directory.")


def subtitle_format(value):
    normalized = value.lower().lstrip('.')
    supported = {extension.lstrip('.') for extension in SUBTITLE_FORMAT_PRIORITY}
    if normalized != "all" and normalized not in supported:
        raise argparse.ArgumentTypeError(
            f"unsupported subtitle format '{value}'; choose from all, {', '.join(sorted(supported))}"
        )
    return normalized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mux anime videos and subtitle files into MKVs, preserving formatting.")
    parser.add_argument("path", help="Path to the directory containing your videos and subtitle files.")
    parser.add_argument(
        "--subtitle-format", "--format",
        type=subtitle_format,
        default="all",
        metavar="FORMAT",
        help="Subtitle format to search for: ass, ssa, srt, vtt, or all (default: all; priority: ASS, SSA, SRT, VTT).",
    )
    args = parser.parse_args()
    
    check_ffmpeg()
    process_directory(args.path, args.subtitle_format)
