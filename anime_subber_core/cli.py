import argparse

from .config import RuntimeConfig
from .pipeline import process_target


def build_parser():
    parser = argparse.ArgumentParser(description="Generate subtitles using Gemini text and Whisper timing.")
    parser.add_argument("path", help="Video/audio file or directory")
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument("--ocr", dest="ocr", action="store_true", help="Enable OCR (default)")
    ocr_group.add_argument("--no-ocr", "--no_ocr", dest="ocr", action="store_false", help="Disable OCR")
    parser.set_defaults(ocr=True)
    parser.add_argument("--ocr-only", "--ocr_only", dest="ocr_only", action="store_true")
    parser.add_argument("--strict-timing", "--strict_timing", dest="strict_timing", action="store_true",
                        help="Trim leading quiet audio from contiguous cues")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--force-update", "--force_update", dest="force_update", action="store_true")
    parser.add_argument("--format", choices=("srt", "ass"), default="ass",
                        help="Output format (default: ass)")
    parser.add_argument("--gemini-workers", "--gemini_workers", dest="gemini_workers", type=int, default=4)
    parser.add_argument("--whisper-workers", "--whisper_workers", dest="whisper_workers", type=int, default=1)
    parser.add_argument("--ocr-gpu", "--ocr_gpu", dest="ocr_gpu", action="store_true")
    parser.add_argument("--whisper-model", "--whisper_model", default="large")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    runtime = RuntimeConfig(args.gemini_workers, args.whisper_workers, args.ocr_gpu, args.strict_timing)
    process_target(args.path, args.format, args.ocr, args.ocr_only, args.force_update, args.lite,
                   runtime, args.whisper_model, args.device)
