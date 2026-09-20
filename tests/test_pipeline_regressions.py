import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anime_subber_core.models import Subtitle
from anime_subber_core.pipeline import (_deduplicate_cues, process_target,
                                        process_video)
from anime_subber_core.subtitles import read_ass, write_ass


class PipelineTests(unittest.TestCase):
    def test_missing_target_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                process_target(str(missing))

    def test_spatially_distinct_identical_ocr_signs_are_not_deduplicated(self):
        cues = [
            Subtitle(1.0, 4.0, "[Exit]", x=100.0, y=100.0, layer=1),
            Subtitle(1.1, 4.1, "[Exit]", x=1700.0, y=100.0, layer=1),
        ]
        self.assertEqual(len(_deduplicate_cues(cues, (1920, 1080))), 2)

    def test_overlapping_duplicate_dialogue_is_deduplicated(self):
        cues = [Subtitle(1.0, 3.0, "Hello"), Subtitle(1.1, 3.1, "Hello")]
        self.assertEqual(len(_deduplicate_cues(cues)), 1)

    def test_ocr_only_preserves_existing_ass_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "episode.mkv"
            video.write_bytes(b"placeholder")
            output = root / "episode.ass"
            write_ass(str(output), [Subtitle(1.0, 2.0, "Existing dialogue")])
            with patch("anime_subber_core.pipeline.process_video_signs", return_value=([], (1920, 1080))):
                process_video(str(video), str(output), ocr_only=True)
            restored = read_ass(str(output))
            self.assertEqual([cue.text for cue in restored], ["Existing dialogue"])


if __name__ == "__main__":
    unittest.main()
