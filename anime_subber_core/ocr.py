"""On-screen Japanese text detection and translation (kept separate from audio alignment)."""
import concurrent.futures
import re
from typing import List, Tuple

from .cache import CacheStore
from .gemini import GeminiManager, translate_text_batch
from .models import Subtitle
from .similarity import ratio
from .text import normalize_text


_KANJI = re.compile(r"[一-龯々〆ヵヶ]")
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_LATIN = re.compile(r"[A-Za-z]")


def contains_japanese(text: str) -> bool:
    """Identify substantive Japanese while ignoring punctuation-like OCR noise.

    A single kana next to Latin text is commonly a misread bracket (`くON AIR〉`).
    Kanji, two or more kana, or a kana-only sign still count as Japanese.
    """
    value = text or ""
    if _KANJI.search(value):
        return True
    kana_count = len(_KANA.findall(value))
    latin_count = len(_LATIN.findall(value))
    return kana_count >= 2 or (kana_count == 1 and latin_count == 0)


def _bbox_bounds(bbox):
    xs, ys = [point[0] for point in bbox], [point[1] for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _spatially_nested(first, second, threshold=0.78):
    a1, b1, a2, b2 = _bbox_bounds(first)
    c1, d1, c2, d2 = _bbox_bounds(second)
    intersection = max(0, min(a2, c2) - max(a1, c1)) * max(0, min(b2, d2) - max(b1, d1))
    first_area = max(1, (a2 - a1) * (b2 - b1))
    second_area = max(1, (c2 - c1) * (d2 - d1))
    return intersection / min(first_area, second_area) >= threshold


def _prepare_detections(raw_detections):
    """Drop Latin-only and nested component detections from one OCR frame."""
    candidates = []
    for bbox, text, confidence in raw_detections:
        norm = normalize_text(text)
        if confidence < 0.35 or len(norm) < 2 or not contains_japanese(text):
            continue
        native_bbox = [[float(point[0]), float(point[1])] for point in bbox]
        candidates.append((native_bbox, text, float(confidence), norm))
    candidates.sort(key=lambda item: (len(item[3]), item[2]), reverse=True)
    accepted = []
    for candidate in candidates:
        bbox, _, _, norm = candidate
        nested = any(_spatially_nested(bbox, kept[0]) and (norm in kept[3] or kept[3] in norm)
                     for kept in accepted)
        if not nested:
            accepted.append(candidate)
    return accepted


def _position(bbox, width: float, height: float) -> Tuple[int, float, float]:
    x = sum(p[0] for p in bbox) / len(bbox)
    y = sum(p[1] for p in bbox) / len(bbox)
    col = 1 if x < width / 3 else (3 if x > 2 * width / 3 else 2)
    row = 3 if y < height / 3 else (1 if y > 2 * height / 3 else 2)
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]][row - 1][col - 1], x, y


def _resolution(video_file: str):
    import cv2
    cap = cv2.VideoCapture(video_file)
    try:
        return round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    finally:
        cap.release()


def _frame_has_sign(cap, reader, frame_index, norm_target, bbox):
    """Check one localized frame for a sign during boundary refinement."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return False
    height, width = frame.shape[:2]
    xs, ys = [point[0] for point in bbox], [point[1] for point in bbox]
    x1, x2 = max(0, round(min(xs)) - 24), min(width, round(max(xs)) + 24)
    y1, y2 = max(0, round(min(ys)) - 24), min(height, round(max(ys)) + 24)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame[y1:y2, x1:x2]
    detected = reader.readtext(crop, detail=0)
    return any(ratio(norm_target, normalize_text(text)) >= 78 for text in detected)


def _first_visible_frame(cap, reader, hit_frame, stride, norm_target, bbox):
    low, high, first = max(0, hit_frame - stride + 1), hit_frame, hit_frame
    while low <= high:
        middle = (low + high) // 2
        if _frame_has_sign(cap, reader, middle, norm_target, bbox):
            first, high = middle, middle - 1
        else:
            low = middle + 1
    return first


def _last_visible_frame(cap, reader, last_hit, first_miss, norm_target, bbox):
    low, high, last = last_hit, max(last_hit, first_miss - 1), last_hit
    while low <= high:
        middle = (low + high) // 2
        if _frame_has_sign(cap, reader, middle, norm_target, bbox):
            last, low = middle, middle + 1
        else:
            high = middle - 1
    return last


def detect_signs(video_file: str, cache: CacheStore, gpu=False, sample_seconds=1.0):
    cached = cache.load_json(video_file, "ocr_signs_v4")
    if cached is not None:
        resolution = _resolution(video_file)
        # Original caches stored grouped line bboxes without a group centroid.
        # Enrich those records during migration so ASS can use exact coordinates.
        for sign in cached:
            if "x" not in sign or "y" not in sign:
                boxes = [line.get("bbox") for line in sign.get("lines", []) if line.get("bbox")]
                if not boxes and sign.get("bbox"):
                    boxes = [sign["bbox"]]
                points = [point for box in boxes for point in box]
                if points:
                    sign["x"] = sum(point[0] for point in points) / len(points)
                    sign["y"] = sum(point[1] for point in points) / len(points)
                else:
                    sign["x"], sign["y"] = resolution[0] / 2, resolution[1] / 2
        return cached, resolution
    import cv2
    import easyocr
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width, height = _resolution(video_file)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps * sample_seconds))
    reader = easyocr.Reader(["ja", "en"], gpu=gpu, verbose=False)
    active, finished = [], []
    try:
        for frame_index in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            now = frame_index / fps
            seen = set()
            for bbox, text, confidence, norm in _prepare_detections(reader.readtext(frame)):
                _, detected_x, detected_y = _position(bbox, width, height)
                found = next((i for i, sign in enumerate(active)
                              if (ratio(norm, sign["norm_text"]) >= 82 or
                                  norm in sign["norm_text"] or sign["norm_text"] in norm) and
                              _spatially_nested(bbox, sign["bbox"], threshold=0.55) and
                              abs(detected_x - sign["x"]) <= max(80, width * 0.08) and
                              abs(detected_y - sign["y"]) <= max(60, height * 0.08)), None)
                if found is None:
                    pos, x, y = _position(bbox, width, height)
                    first_frame = _first_visible_frame(cap, reader, frame_index, stride, norm, bbox)
                    active.append({"start": first_frame / fps, "last_seen": now,
                                   "last_seen_frame": frame_index, "ja_text": text, "norm_text": norm,
                                   "bbox": bbox, "pos": pos, "x": x, "y": y})
                    found = len(active) - 1
                else:
                    if len(norm) > len(active[found]["norm_text"]):
                        active[found].update({"ja_text": text, "norm_text": norm, "bbox": bbox,
                                              "x": detected_x, "y": detected_y})
                    active[found]["last_seen"] = now
                    active[found]["last_seen_frame"] = frame_index
                seen.add(found)
            for index in reversed(range(len(active))):
                if index not in seen and now - active[index]["last_seen"] >= sample_seconds:
                    sign = active.pop(index)
                    last_frame = _last_visible_frame(cap, reader, sign["last_seen_frame"], frame_index,
                                                     sign["norm_text"], sign["bbox"])
                    sign["end"] = min(total / fps, (last_frame + 1) / fps)
                    finished.append(sign)
    finally:
        cap.release()
    duration = total / fps
    if active:
        boundary_cap = cv2.VideoCapture(video_file)
        try:
            for sign in active:
                last_frame = _last_visible_frame(boundary_cap, reader, sign["last_seen_frame"], total,
                                                 sign["norm_text"], sign["bbox"])
                sign["end"] = min(duration, (last_frame + 1) / fps)
                finished.append(sign)
        finally:
            boundary_cap.release()
    # A final containment pass catches a component that was intermittently
    # recognized as a separate sign on different sampled frames.
    deduplicated = []
    for sign in sorted(finished, key=lambda item: (-len(item["norm_text"]), item["start"])):
        parent = next((kept for kept in deduplicated
                       if min(sign["end"], kept["end"]) - max(sign["start"], kept["start"]) > 0.05 and
                       _spatially_nested(sign["bbox"], kept["bbox"]) and
                       (sign["norm_text"] in kept["norm_text"] or kept["norm_text"] in sign["norm_text"])), None)
        if parent:
            parent["start"] = min(parent["start"], sign["start"])
            parent["end"] = max(parent["end"], sign["end"])
        else:
            deduplicated.append(sign)
    finished = sorted(deduplicated, key=lambda item: item["start"])
    cache.save_json(video_file, "ocr_signs_v4", finished)
    return finished, (width, height)


def translate_signs(signs, video_file: str, manager: GeminiManager, cache: CacheStore,
                    executor: concurrent.futures.Executor) -> List[Subtitle]:
    signs = [sign for sign in signs if contains_japanese(sign.get("ja_text", ""))]
    def translate(index, batch):
        name = f"gemini_ocr_v4_batch_{index}"
        data = cache.load_json(video_file, name)
        if not data:
            data = translate_text_batch([{"id": i, "ja": sign["ja_text"]} for i, sign in enumerate(batch)], manager)
            if data:
                cache.save_json(video_file, name, data)
        mapping = {item.get("id"): item.get("en", "") for item in data if isinstance(item, dict)}
        return [Subtitle(float(sign["start"]), float(sign["end"]), f"[{mapping[i]}]",
                         int(sign["pos"]), float(sign["x"]), float(sign["y"]), 1)
                for i, sign in enumerate(batch) if mapping.get(i)]

    batches = [signs[i:i + 50] for i in range(0, len(signs), 50)]
    futures = [executor.submit(translate, i, batch) for i, batch in enumerate(batches)]
    return [cue for future in futures for cue in future.result()]


def process_video_signs(video_file, manager, cache, executor, gpu=False):
    signs, resolution = detect_signs(video_file, cache, gpu)
    return translate_signs(signs, video_file, manager, cache, executor), resolution
