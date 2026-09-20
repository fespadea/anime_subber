import unittest

from anime_subber_core.ocr import (_prepare_detections, _read_frame_ocr,
                                   _vertical_rescue_regions)


def _box(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


class VerticalOcrTests(unittest.TestCase):
    def test_vertical_fragments_are_read_top_to_bottom(self):
        detections = [
            (_box(100, 80, 130, 110), "語", 0.94),
            (_box(100, 10, 130, 40), "日", 0.96),
            (_box(100, 45, 130, 75), "本", 0.95),
        ]
        prepared = _prepare_detections(detections)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0][1], "日本語")
        self.assertEqual(prepared[0][4], "vertical")

    def test_vertical_columns_are_read_right_to_left(self):
        detections = [
            (_box(60, 45, 90, 75), "文", 0.91),
            (_box(110, 45, 140, 75), "本", 0.95),
            (_box(60, 10, 90, 40), "語", 0.92),
            (_box(110, 10, 140, 40), "日", 0.96),
        ]
        prepared = _prepare_detections(detections)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0][1], "日本\n語文")
        self.assertEqual(prepared[0][3], "日本語文")
        self.assertEqual(prepared[0][4], "vertical")

    def test_distant_vertical_columns_are_not_merged(self):
        detections = [
            (_box(20, 10, 50, 40), "日", 0.95),
            (_box(20, 45, 50, 75), "本", 0.95),
            (_box(300, 10, 330, 40), "語", 0.95),
            (_box(300, 45, 330, 75), "文", 0.95),
        ]
        prepared = _prepare_detections(detections)
        self.assertEqual([item[1] for item in prepared], ["日本", "語文"])
        self.assertTrue(all(item[4] == "vertical" for item in prepared))

    def test_horizontal_japanese_is_preserved(self):
        detections = [(_box(20, 200, 180, 235), "こんにちは", 0.98)]
        prepared = _prepare_detections(detections)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0][1], "こんにちは")
        self.assertEqual(prepared[0][4], "horizontal")

    def test_low_confidence_non_japanese_fragments_can_seed_vertical_rescue(self):
        # Recognition is intentionally garbage/weak. Geometry should survive so
        # Gemini vision can read the crop instead of losing it before grouping.
        raw = [
            (_box(100, 10, 130, 40), "l", 0.08),
            (_box(101, 45, 131, 75), "?", 0.06),
            (_box(99, 80, 129, 110), "I", 0.09),
        ]
        self.assertEqual(_prepare_detections(raw), [])
        rescue = _vertical_rescue_regions(raw, [], 1920, 1080)
        self.assertEqual(len(rescue), 1)
        self.assertLessEqual(rescue[0][0][0], 100)
        self.assertGreaterEqual(rescue[0][2][1], 110)

    def test_frame_ocr_uses_recall_oriented_easyocr_settings(self):
        class Image:
            shape = (1080, 1920, 3)

        class Reader:
            def __init__(self):
                self.kwargs = None

            def readtext(self, _image, **kwargs):
                self.kwargs = kwargs
                return [
                    (_box(100, 10, 130, 40), "l", 0.08),
                    (_box(100, 45, 130, 75), "?", 0.08),
                ]

        reader = Reader()
        recognized, rescue = _read_frame_ocr(reader, Image())
        self.assertEqual(recognized, [])
        self.assertEqual(len(rescue), 1)
        self.assertEqual(reader.kwargs["text_threshold"], 0.50)
        self.assertEqual(reader.kwargs["low_text"], 0.25)
        self.assertEqual(reader.kwargs["link_threshold"], 0.25)
        self.assertEqual(reader.kwargs["canvas_size"], 3840)
        self.assertEqual(reader.kwargs["mag_ratio"], 1.50)


if __name__ == "__main__":
    unittest.main()
