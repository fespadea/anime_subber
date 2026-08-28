import json
import tempfile
import unittest
from pathlib import Path

from anime_subber_core.cache import CacheStore
from anime_subber_core.models import Subtitle
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


if __name__ == "__main__":
    unittest.main()
