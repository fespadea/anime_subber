def refine_contiguous_starts(cues, audio):
    """Trim leading quiet audio for cues at zero or touching the previous cue."""
    previous_end = 0.0
    for cue in sorted((item for item in cues if item.layer == 0), key=lambda item: item.start):
        if cue.start <= 0.001 or abs(cue.start - previous_end) <= 0.01:
            duration = min(1.0, cue.end - cue.start)
            if duration > 0.1:
                segment = audio[round(cue.start * 1000):round((cue.start + duration) * 1000)]
                peak = segment.max_dBFS
                if peak != float("-inf"):
                    for offset in range(0, len(segment), 50):
                        if segment[offset:offset + 50].max_dBFS > peak - 15:
                            cue.start += offset / 1000.0
                            break
        previous_end = cue.end
    return cues
