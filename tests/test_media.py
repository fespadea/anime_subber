import tempfile
import unittest
from pathlib import Path

from anime_subber_core import media
from anime_subber_core.cache import CacheStore
from anime_subber_core.media import (_audio_command, _gemini_remux_command,
                                      _gemini_transcode_command)


class MediaTests(unittest.TestCase):
    def test_audio_extraction_is_bounded_audio_only_pcm(self):
        command = _audio_command("episode.webm", "audio.wav")
        self.assertIn("-nostdin", command)
        self.assertIn("-vn", command)
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")

    def test_gemini_proxy_prefers_video_stream_copy_with_aac_audio(self):
        command = _gemini_remux_command("episode.mkv", "proxy.mp4")
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertIn("0:v:0", command)

    def test_gemini_proxy_has_h264_fallback(self):
        command = _gemini_transcode_command("episode.mkv", "proxy.mp4")
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")

    def test_supported_mov_uses_documented_gemini_mime_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "clip.mov"
            source.write_bytes(b"video")
            cache = CacheStore(root=Path(tmpdir) / "cache")
            path, mime = media.prepare_gemini_video(str(source), cache)
            self.assertEqual(Path(path), source)
            self.assertEqual(mime, "video/mov")


if __name__ == "__main__":
    unittest.main()
