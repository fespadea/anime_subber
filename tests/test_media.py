import unittest

from anime_subber_core.media import _audio_command


class MediaTests(unittest.TestCase):
    def test_audio_extraction_is_bounded_audio_only_pcm(self):
        command = _audio_command("episode.webm", "audio.wav")
        self.assertIn("-nostdin", command)
        self.assertIn("-vn", command)
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")


if __name__ == "__main__":
    unittest.main()
