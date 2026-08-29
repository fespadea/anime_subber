import unittest

from anime_subber_core.pipeline import _merge_recovery_segments
from anime_subber_core.whisper_engine import _transcribe


class WhisperRecoveryTests(unittest.TestCase):
    def test_full_transcription_uses_anti_hallucination_options(self):
        class Model:
            def __init__(self):
                self.options = None

            def transcribe(self, _path, **options):
                self.options = options
                return {"segments": []}

        model = Model()
        _transcribe(model, "audio.wav")
        self.assertFalse(model.options["condition_on_previous_text"])
        self.assertEqual(model.options["temperature"], 0)
        self.assertTrue(model.options["word_timestamps"])
        self.assertEqual(model.options["hallucination_silence_threshold"], 2.0)

    def test_targeted_segments_replace_hallucinated_span(self):
        original = [
            {"start": 0, "end": 20, "text": "valid"},
            {"start": 20, "end": 50, "text": "おめでとう"},
            {"start": 50, "end": 60, "text": "valid again"},
        ]
        replacements = [{"start": 20, "end": 35, "text": "recovered dialogue"}]
        merged = _merge_recovery_segments(original, replacements, 20, 50)
        self.assertEqual([segment["text"] for segment in merged],
                         ["valid", "recovered dialogue", "valid again"])


if __name__ == "__main__":
    unittest.main()
