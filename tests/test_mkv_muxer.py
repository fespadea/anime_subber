import tempfile
import unittest
from pathlib import Path

from mkv_muxer import find_subtitle, subtitle_format


class MkvMuxerTests(unittest.TestCase):
    def test_all_formats_prioritizes_ass_over_srt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode.srt").touch()
            (root / "episode.ass").touch()
            self.assertEqual(Path(find_subtitle(temporary, "episode")).suffix, ".ass")

    def test_explicit_format_restricts_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode.ass").touch()
            (root / "episode.srt").touch()
            self.assertEqual(Path(find_subtitle(temporary, "episode", "srt")).suffix, ".srt")

    def test_format_accepts_leading_dot(self):
        self.assertEqual(subtitle_format(".ASS"), "ass")


if __name__ == "__main__":
    unittest.main()
