import os
import tempfile
import unittest
from pathlib import Path

from anime_subber_core.cache import CacheStore


class CacheTests(unittest.TestCase):
    def test_replacing_media_at_same_path_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "episode.mkv"
            media.write_bytes(b"first")
            cache = CacheStore(root / "cache")
            cache.save_json(str(media), "sample", {"value": 1})
            first_dir = cache.media_dir(str(media))
            self.assertEqual(cache.load_json(str(media), "sample"), {"value": 1})

            old_stat = media.stat()
            media.write_bytes(b"replacement-media")
            os.utime(media, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
            second_dir = cache.media_dir(str(media))

            self.assertNotEqual(first_dir, second_dir)
            self.assertIsNone(cache.load_json(str(media), "sample"))


if __name__ == "__main__":
    unittest.main()
