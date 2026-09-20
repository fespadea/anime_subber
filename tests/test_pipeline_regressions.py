import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anime_subber_core.models import Subtitle
from anime_subber_core.pipeline import (_deduplicate_cues, _same_ocr_sign,
                                        process_target, process_video)
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

    def test_retranslated_same_ocr_sign_is_deduplicated_by_time_and_position(self):
        cues = [
            Subtitle(21.23, 25.53, "[I wanted to thank Lip and return the money]",
                     x=580.0, y=255.0, layer=1),
            Subtitle(21.23, 25.53, "[I wanted to say thanks and pay for the lipstick]",
                     x=580.0, y=227.0, layer=1),
        ]
        self.assertEqual(len(_deduplicate_cues(cues, (1920, 1080))), 1)

    def test_ocr_provenance_matches_even_when_translation_changes(self):
        old = Subtitle(1.0, 4.0, "[Old wording]", x=500, y=300, layer=1,
                       effect="anime_subber_ocr:abc")
        new = Subtitle(1.1, 4.1, "[Completely different wording]", x=530, y=325, layer=1,
                       effect="anime_subber_ocr:abc")
        self.assertTrue(_same_ocr_sign(old, new, (1920, 1080)))

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

    def test_ocr_only_keeps_existing_sign_instead_of_adding_retranslation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "episode.mkv"
            video.write_bytes(b"placeholder")
            output = root / "episode.ass"
            existing = Subtitle(21.23, 25.53, "[I wanted to thank Lip and return the money]",
                                x=580.0, y=241.0, layer=1)
            replacement = Subtitle(21.23, 25.53, "[I wanted to say thanks and pay for the lipstick]",
                                   x=580.0, y=241.0, layer=1)
            write_ass(str(output), [existing])
            with patch("anime_subber_core.pipeline.process_video_signs",
                       return_value=([replacement], (1920, 1080))):
                process_video(str(video), str(output), ocr_only=True)
            restored = read_ass(str(output))
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].text, existing.text)

    def test_gemini_input_bypasses_api_video_transcription(self):
        import json

        class FakeAudio:
            def __len__(self):
                return 10_000

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "episode.mkv"
            video.write_bytes(b"placeholder")
            gem = root / "gemini.json"
            gem.write_text(json.dumps({
                "format": "anime-subber-gem-v1",
                "cues": [{
                    "kind": "dialogue", "start": 1.0, "end": 2.5,
                    "ja": "こんにちは", "en": "Hello."
                }]
            }, ensure_ascii=False), encoding="utf-8")
            output = root / "episode.ass"
            whisper = [{"text": "こんにちは", "start": 1.2, "end": 2.0, "global_idx": 0}]
            with patch("anime_subber_core.pipeline.video_resolution", return_value=(1920, 1080)), \
                 patch("anime_subber_core.pipeline.load_audio", return_value=FakeAudio()), \
                 patch("anime_subber_core.pipeline.safe_instance_count", return_value=1), \
                 patch("anime_subber_core.pipeline.load_model", return_value=object()), \
                 patch("anime_subber_core.pipeline.transcribe_full", return_value=whisper), \
                 patch("anime_subber_core.pipeline.prepare_gemini_video") as prepare_video:
                process_video(str(video), str(output), run_ocr=False, gemini_input=str(gem))
            prepare_video.assert_not_called()
            restored = read_ass(str(output))
            self.assertEqual([cue.text for cue in restored], ["Hello."])
            self.assertAlmostEqual(restored[0].start, 1.2, places=1)


if __name__ == "__main__":
    unittest.main()
