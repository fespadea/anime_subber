import argparse

from .config import RuntimeConfig
from .pipeline import process_target


def build_parser():
    parser = argparse.ArgumentParser(description="Generate subtitles using Gemini text and Whisper timing.")
    parser.add_argument("path", help="Video/audio file or directory")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr_only", action="store_true")
    parser.add_argument("--strict_timing", action="store_true", help="Trim leading quiet audio from contiguous cues")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--force_update", action="store_true")
    parser.add_argument("--format", choices=("srt", "ass"), default="srt",
                        help="ASS supports precise OCR placement (default: srt)")
    parser.add_argument("--gemini-workers", type=int, default=4)
    parser.add_argument("--whisper-workers", type=int, default=1)
    parser.add_argument("--ocr-gpu", action="store_true")
    parser.add_argument("--whisper-model", default="large")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    runtime = RuntimeConfig(args.gemini_workers, args.whisper_workers, args.ocr_gpu, args.strict_timing)
    process_target(args.path, args.format, args.ocr, args.ocr_only, args.force_update, args.lite,
                   runtime, args.whisper_model, args.device)
