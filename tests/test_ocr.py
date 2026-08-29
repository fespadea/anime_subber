import unittest
from unittest.mock import patch

from anime_subber_core.ocr import (_first_visible_frame, _last_visible_frame,
                                   _prepare_detections, contains_japanese)


class OcrTimingTests(unittest.TestCase):
    def test_latin_only_signs_are_skipped_but_mixed_signs_remain(self):
        self.assertFalse(contains_japanese("BS11: Oct 9"))
        self.assertFalse(contains_japanese("くON AIR〉"))
        self.assertFalse(contains_japanese("〈ON AIR〉"))
        self.assertTrue(contains_japanese("BS11 アニメ Oct 9"))
        self.assertTrue(contains_japanese("愛 LOVE"))
        self.assertTrue(contains_japanese("あ"))
        bbox = [[0, 0], [100, 0], [100, 30], [0, 30]]
        detections = _prepare_detections([(bbox, "BS11: Oct 9", .99), (bbox, "BS11 アニメ", .90)])
        self.assertEqual([item[1] for item in detections], ["BS11 アニメ"])

    def test_nested_component_detections_keep_only_complete_japanese_sign(self):
        full = [[0, 0], [200, 0], [200, 40], [0, 40]]
        left = [[0, 0], [90, 0], [90, 40], [0, 40]]
        right = [[100, 0], [200, 0], [200, 40], [100, 40]]
        detections = _prepare_detections([
            (full, "放送開始予定", .88), (left, "放送開始", .95), (right, "予定", .96)
        ])
        self.assertEqual([item[1] for item in detections], ["放送開始予定"])

    def test_binary_search_refines_first_and_last_visible_frames(self):
        visible = lambda _cap, _reader, frame, _text, _bbox: 3 <= frame <= 12
        with patch("anime_subber_core.ocr._frame_has_sign", side_effect=visible):
            self.assertEqual(_first_visible_frame(None, None, 8, 8, "title", []), 3)
            self.assertEqual(_last_visible_frame(None, None, 8, 16, "title", []), 12)


if __name__ == "__main__":
    unittest.main()
