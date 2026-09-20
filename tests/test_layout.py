import unittest

from anime_subber_core.cli import build_parser
from anime_subber_core.layout import _inside_screen, _rect, resolve_collisions
from anime_subber_core.models import Subtitle


class LayoutTests(unittest.TestCase):
    def test_ocr_moves_away_from_overlapping_dialogue(self):
        dialogue = Subtitle(1, 4, "Ordinary dialogue at the bottom")
        sign = Subtitle(2, 3, "[A sign]", 2, 960, 1000, 1)
        original = (sign.x, sign.y)
        resolve_collisions([dialogue, sign], (1920, 1080))
        self.assertLess(sign.font_size, 46)
        self.assertNotEqual((sign.x, sign.y), original)
        self.assertFalse(_rect(dialogue, (1920, 1080)).overlaps(_rect(sign, (1920, 1080))))

    def test_both_overlapping_ocr_cues_shrink_and_separate(self):
        first = Subtitle(1, 5, "[First sign]", 5, 640, 360, 1)
        second = Subtitle(2, 4, "[Second sign]", 5, 640, 360, 1)
        resolve_collisions([first, second], (1280, 720))
        self.assertLess(first.font_size, 46)
        self.assertLess(second.font_size, 46)
        self.assertAlmostEqual((first.x + second.x) / 2, 640, delta=1)
        self.assertAlmostEqual((first.y + second.y) / 2, 360, delta=1)
        self.assertFalse(_rect(first, (1280, 720)).overlaps(_rect(second, (1280, 720))))

    def test_ass_and_ocr_are_defaults_with_opt_out(self):
        defaults = build_parser().parse_args(["episode.mkv"])
        self.assertEqual(defaults.format, "ass")
        self.assertTrue(defaults.ocr)
        self.assertFalse(defaults.ocr_vision_rescue)
        rescue = build_parser().parse_args(["episode.mkv", "--ocr-vision-rescue"])
        self.assertTrue(rescue.ocr_vision_rescue)
        opted_out = build_parser().parse_args(["episode.mkv", "--no-ocr", "--format", "srt"])
        self.assertFalse(opted_out.ocr)
        self.assertEqual(opted_out.format, "srt")

    def test_offscreen_ocr_shrinks_then_moves_inside_frame(self):
        sign = Subtitle(1, 2, "[A very long translated Japanese sign]", 7, 2, 2, 1)
        resolve_collisions([sign], (640, 360))
        self.assertLess(sign.font_size, 46)
        self.assertTrue(_inside_screen(sign, (640, 360)))


if __name__ == "__main__":
    unittest.main()
