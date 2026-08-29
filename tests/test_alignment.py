import unittest
from unittest.mock import patch

from anime_subber_core.alignment import align_subtitles, global_monotonic_matches


class AlignmentTests(unittest.TestCase):
    def test_missed_early_line_does_not_cascade_to_tail(self):
        gemini = [
            {"ja": "存在しない台詞", "en": "A missed line"},
            {"ja": "おはようございます", "en": "Good morning."},
            {"ja": "また明日", "en": "See you tomorrow."},
        ]
        whisper = [
            {"start": 10.0, "end": 11.0, "text": "おはようございます", "global_idx": 0},
            {"start": 50.0, "end": 51.0, "text": "また明日", "global_idx": 1},
        ]
        cues, used, _ = align_subtitles(gemini, whisper, 0.0, 60.0)
        by_text = {cue.text: cue for cue in cues}
        self.assertAlmostEqual(by_text["Good morning."].start, 10.0)
        self.assertAlmostEqual(by_text["See you tomorrow."].start, 50.0)
        self.assertLess(by_text["A missed line"].end, 10.01)
        self.assertEqual(used, {0, 1})

    def test_reverse_anchor_bounds_trailing_misses(self):
        gemini = [
            {"ja": "最初", "en": "First"},
            {"ja": "欠落一", "en": "Missing one"},
            {"ja": "欠落二", "en": "Missing two"},
            {"ja": "最後の言葉", "en": "Last words"},
        ]
        whisper = [
            {"start": 2.0, "end": 3.0, "text": "最初"},
            {"start": 42.0, "end": 43.0, "text": "最後の言葉"},
        ]
        cues, _, _ = align_subtitles(gemini, whisper, 0.0, 45.0)
        self.assertEqual(next(c for c in cues if c.text == "Last words").start, 42.0)
        self.assertTrue(all(0 <= c.start < c.end <= 45 for c in cues))

    def test_matches_are_monotonic(self):
        lines = [{"ja": value, "en": value} for value in ("甲です", "乙です", "丙です")]
        segments = [{"text": value, "start": i, "end": i + .5} for i, value in enumerate(("甲です", "雑音", "乙です", "丙です"))]
        matches = global_monotonic_matches(lines, segments)
        self.assertEqual([m.whisper_start for m in matches], [0, 2, 3])

    def test_repeated_text_inside_one_whisper_segment_is_consumed_once(self):
        gemini = [
            {"ja": "運命の", "en": "Fate..."},
            {"ja": "運命。", "en": "Fate."},
        ]
        whisper = [{"start": 17.68, "end": 24.02, "text": "運命の 運命", "global_idx": 5}]
        cues, used, recovery = align_subtitles(gemini, whisper, 17.68, 24.02)
        self.assertEqual([cue.text for cue in cues], ["Fate...", "Fate."])
        self.assertLessEqual(cues[0].end, cues[1].start)
        self.assertEqual(used, {5})
        self.assertEqual(recovery, [])

    def test_shared_suffix_prefix_requires_two_source_occurrences(self):
        gemini = [
            {"ja": "宝はほかでもない", "en": "The treasure is none other than..."},
            {"ja": "ほかでもない友達", "en": "None other than our friends."},
        ]
        whisper = [{"start": 0, "end": 10, "text": "宝はほかでもない、ほかでもない友達"}]
        matches = global_monotonic_matches(gemini, whisper)
        self.assertEqual(len(matches), 2)
        self.assertLessEqual(matches[0].character_end, matches[1].character_start)

    def test_phonetic_homophones_map_to_separate_occurrences(self):
        gemini = [
            {"ja": "シュコー", "en": "Huuuh..."},
            {"ja": "シュコー", "en": "Puuuh..."},
        ]
        whisper = [{"start": 0.64, "end": 3.44, "text": "出航 出航"}]

        def reading(text):
            return text.replace("シュコー", "しゅこう").replace("出航", "しゅこう")

        with patch("anime_subber_core.alignment.phonetic_text", side_effect=reading):
            matches = global_monotonic_matches(gemini, whisper)
        self.assertEqual(len(matches), 2)
        self.assertAlmostEqual(matches[0].start_time, 0.64)
        self.assertAlmostEqual(matches[-1].end_time, 3.44)
        self.assertLessEqual(matches[0].character_end, matches[1].character_start)


if __name__ == "__main__":
    unittest.main()
