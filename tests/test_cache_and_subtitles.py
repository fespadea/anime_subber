import json
import tempfile
import unittest
from pathlib import Path

from anime_subber_core.cache import CacheStore
from anime_subber_core.models import Subtitle
from anime_subber_core.styling import dialogue_font_size
from anime_subber_core.subtitles import write_ass


class CacheAndSubtitleTests(unittest.TestCase):
    def test_cache_namespaces_equal_stems_by_full_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CacheStore(Path(temporary))
            self.assertNotEqual(cache.media_dir("a/episode.mkv"), cache.media_dir("b/episode.mkv"))
            path = cache.save_json("a/episode.mkv", "whisper", [{"text": "ok"}])
            self.assertTrue(path.is_relative_to(Path(temporary)))
            self.assertEqual(cache.load_json("a/episode.mkv", "whisper")[0]["text"], "ok")

    def test_ass_uses_exact_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.ass"
            write_ass(str(path), [Subtitle(1, 2, "Sign", 5, 321.4, 123.6, 1)], (1280, 720))
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn(r"\pos(321,124)", text)
            self.assertIn("PlayResX: 1280", text)
            self.assertIn(f"Style: Dialogue,Arial,{dialogue_font_size((1280, 720)):.2f},", text)

    def test_ass_writes_per_cue_font_size_with_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.ass"
            write_ass(str(path), [Subtitle(1, 2, "Sign", 5, 321, 124, 1, 32)], (1280, 720))
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn(r"{\pos(321,124)\fs32}", text)

    def test_ass_dialogue_style_scales_with_video_height(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "480p.ass"
            write_ass(str(path), [Subtitle(1, 2, "Dialogue")], (854, 480))
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn("PlayResY: 480", text)
            self.assertIn(f"Style: Dialogue,Arial,{dialogue_font_size((854, 480)):.2f},", text)

    def test_cache_serializes_numpy_style_scalars_and_arrays(self):
        class Scalar:
            def item(self):
                return 42

        class Array:
            def tolist(self):
                return [[1, 2], [3, 4]]

        with tempfile.TemporaryDirectory() as temporary:
            cache = CacheStore(Path(temporary))
            cache.save_json("episode.mkv", "ocr_signs", {"coordinate": Scalar(), "bbox": Array()})
            self.assertEqual(cache.load_json("episode.mkv", "ocr_signs"),
                             {"coordinate": 42, "bbox": [[1, 2], [3, 4]]})


if __name__ == "__main__":
    unittest.main()
