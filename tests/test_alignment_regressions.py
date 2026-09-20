import unittest

from anime_subber_core.alignment import (_candidates, _character_timeline,
                                         align_subtitles)


class AlignmentTests(unittest.TestCase):
    def test_character_timeline_prefers_whisper_word_timestamps(self):
        segments = [{
            "text": "日本語",
            "start": 0.0,
            "end": 9.0,
            "words": [
                {"word": "日本", "start": 1.0, "end": 3.0},
                {"word": "語", "start": 7.0, "end": 8.0},
            ],
        }]
        timeline = _character_timeline(segments)
        self.assertEqual("".join(item.value for item in timeline), "日本語")
        self.assertAlmostEqual(timeline[0].start, 1.0)
        self.assertAlmostEqual(timeline[1].end, 3.0)
        self.assertAlmostEqual(timeline[2].start, 7.0)
        self.assertAlmostEqual(timeline[2].end, 8.0)


    def test_character_timeline_falls_back_when_word_text_disagrees(self):
        segments = [{
            "text": "日本語",
            "start": 0.0,
            "end": 9.0,
            "words": [
                {"word": "別物", "start": 7.0, "end": 8.0},
            ],
        }]
        timeline = _character_timeline(segments)
        self.assertEqual("".join(item.value for item in timeline), "日本語")
        self.assertAlmostEqual(timeline[0].start, 0.0)
        self.assertAlmostEqual(timeline[-1].end, 9.0)

    def test_max_span_is_enforced(self):
        line = {"ja": "日本", "en": "Japan"}
        segments = [
            {"text": "日", "start": 0.0, "end": 1.0},
            {"text": "本", "start": 1.0, "end": 2.0},
        ]
        timeline = _character_timeline(segments)
        one_segment = _candidates(0, line, timeline, max_span=1)
        two_segments = _candidates(0, line, timeline, max_span=2)
        self.assertTrue(one_segment)
        self.assertTrue(all(match.whisper_start == match.whisper_end for match in one_segment))
        self.assertTrue(any(match.whisper_start != match.whisper_end for match in two_segments))

    def test_early_missed_line_does_not_push_later_match_to_chunk_end(self):
        lines = [
            {"ja": "これは聞こえない台詞", "en": "This line is missed."},
            {"ja": "こんにちは", "en": "Hello."},
        ]
        segments = [{"text": "こんにちは", "start": 10.0, "end": 12.0}]
        cues, _, recovery = align_subtitles(lines, segments, 0.0, 20.0)
        hello = next(cue for cue in cues if cue.text == "Hello.")
        self.assertLess(hello.start, 12.1)
        self.assertLessEqual(hello.end, 12.1)
        self.assertTrue(recovery)


if __name__ == "__main__":
    unittest.main()
