import unittest

from anime_subber_core.text import normalize_text, parse_llm_json


class TextNormalizationTests(unittest.TestCase):
    def test_unicode_punctuation_and_spacing_do_not_affect_matching(self):
        self.assertEqual(normalize_text("「こんにちは……！」"), "こんにちは")

    def test_nfkc_folds_full_width_latin_and_digits(self):
        self.assertEqual(normalize_text("ＡＢＣ１２３"), "abc123")

    def test_japanese_prolonged_sound_mark_is_preserved(self):
        self.assertEqual(normalize_text("スーパー"), "スーパー")

    def test_json_parser_distinguishes_empty_array_from_invalid_json(self):
        self.assertEqual(parse_llm_json("[]"), [])
        self.assertIsNone(parse_llm_json("not json"))


if __name__ == "__main__":
    unittest.main()
