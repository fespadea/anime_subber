import unittest
from unittest.mock import patch

from anime_subber_core.gemini import (GeminiManager, _audio_lines,
                                      _translation_items, _vision_items)


class GeminiValidationTests(unittest.TestCase):
    def test_audio_lines_reject_wrong_shape(self):
        self.assertEqual(_audio_lines({"ja": "日本語", "en": "Japanese"}), [])
        self.assertEqual(_audio_lines([{"ja": "日本語", "en": "Japanese"}, "bad"]),
                         [{"ja": "日本語", "en": "Japanese"}])

    def test_translation_ids_are_normalized(self):
        self.assertEqual(_translation_items([{"id": "2", "en": "Station"}]),
                         [{"id": 2, "en": "Station"}])

    def test_vision_ocr_ids_are_normalized_and_empty_items_are_ignored(self):
        self.assertEqual(
            _vision_items([{"id": "2", "ja": "私達"}, {"id": 3, "ja": ""}, "bad"]),
            [{"id": 2, "ja": "私達"}],
        )

    def test_transient_rate_limit_is_retried_before_model_is_dropped(self):
        class Models:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("429 rate limit")
                return "ok"

        class Client:
            def __init__(self):
                self.models = Models()

        client = Client()
        manager = GeminiManager(client_factory=lambda: client)
        with patch("anime_subber_core.gemini.time.sleep"):
            result = manager.generate([], object())
        self.assertEqual(result, "ok")
        self.assertEqual(client.models.calls, 2)
        self.assertFalse(manager.api_exhausted)


if __name__ == "__main__":
    unittest.main()
