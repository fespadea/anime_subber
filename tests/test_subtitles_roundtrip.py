import tempfile
import unittest
from pathlib import Path

from anime_subber_core.models import Subtitle
from anime_subber_core.subtitles import read_ass, write_ass


class AssSubtitleTests(unittest.TestCase):
    def test_generated_ass_round_trips_for_ocr_only_updates(self):
        cues = [
            Subtitle(1.25, 3.5, "Dialogue, with {braces} and \\ slash\nsecond line"),
            Subtitle(2.0, 4.0, "[Station]", position=9, x=1550.0, y=120.0,
                     layer=1, font_size=36.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.ass"
            write_ass(str(path), cues, (1920, 1080))
            restored = read_ass(str(path))

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].text, cues[0].text)
        self.assertAlmostEqual(restored[0].start, 1.25, places=2)
        self.assertEqual(restored[1].text, "[Station]")
        self.assertEqual(restored[1].layer, 1)
        self.assertEqual((restored[1].x, restored[1].y), (1550.0, 120.0))
        self.assertEqual(restored[1].font_size, 36.0)


if __name__ == "__main__":
    unittest.main()
