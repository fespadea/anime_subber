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


# OCR cache versions are intentionally tied to the detection/layout algorithm.
# v5 adds vertical Japanese reconstruction and right-to-left column ordering.
_OCR_CACHE_VERSION = "v5"


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


def _bbox_metrics(bbox):
    left, top, right, bottom = _bbox_bounds(bbox)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": width,
        "height": height,
        "cx": (left + right) / 2.0,
        "cy": (top + bottom) / 2.0,
    }


def _enclosing_bbox(boxes):
    bounds = [_bbox_bounds(box) for box in boxes]
    left = min(item[0] for item in bounds)
    top = min(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    bottom = max(item[3] for item in bounds)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _spatially_nested(first, second, threshold=0.78):
    a1, b1, a2, b2 = _bbox_bounds(first)
    c1, d1, c2, d2 = _bbox_bounds(second)
    intersection = max(0, min(a2, c2) - max(a1, c1)) * max(0, min(b2, d2) - max(b1, d1))
    first_area = max(1, (a2 - a1) * (b2 - b1))
    second_area = max(1, (c2 - c1) * (d2 - d1))
    return intersection / min(first_area, second_area) >= threshold


def _horizontal_overlap(first, second):
    left = max(first["left"], second["left"])
    right = min(first["right"], second["right"])
    return max(0.0, right - left) / max(1.0, min(first["width"], second["width"]))


def _vertical_overlap(first, second):
    top = max(first["top"], second["top"])
    bottom = min(first["bottom"], second["bottom"])
    return max(0.0, bottom - top) / max(1.0, min(first["height"], second["height"]))


def _union_groups(items, compatible):
    """Return connected components under a permissive spatial compatibility test."""
    parents = list(range(len(items)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if compatible(items[left], items[right]):
                union(left, right)

    grouped = {}
    for index, item in enumerate(items):
        grouped.setdefault(find(index), []).append(item)
    return list(grouped.values())


def _vertical_fragment_pair(first, second):
    """Whether two OCR fragments plausibly continue the same vertical column."""
    one, two = _bbox_metrics(first[0]), _bbox_metrics(second[0])
    upper, lower = (one, two) if one["cy"] <= two["cy"] else (two, one)

    # Same-column glyphs should be substantially separated vertically. This
    # prevents adjacent characters on a normal horizontal line from clustering.
    center_dy = lower["cy"] - upper["cy"]
    if center_dy < max(3.0, min(one["height"], two["height"]) * 0.38):
        return False

    width_ratio = max(one["width"], two["width"]) / max(1.0, min(one["width"], two["width"]))
    if width_ratio > 2.8:
        return False

    x_distance = abs(one["cx"] - two["cx"])
    aligned = (_horizontal_overlap(one, two) >= 0.24 or
               x_distance <= max(10.0, max(one["width"], two["width"]) * 0.62))
    if not aligned:
        return False

    gap = lower["top"] - upper["bottom"]
    return gap <= max(12.0, max(one["height"], two["height"]) * 1.25)


def _looks_like_vertical_column(group):
    if not group:
        return False
    box = _enclosing_bbox([item[0] for item in group])
    metrics = _bbox_metrics(box)
    if len(group) == 1:
        # EasyOCR occasionally returns an entire vertical line as one tall box.
        return metrics["height"] >= metrics["width"] * 1.35 and len(group[0][3]) >= 2
    return metrics["height"] >= metrics["width"] * 1.18


def _make_vertical_column(group):
    ordered = sorted(group, key=lambda item: (_bbox_metrics(item[0])["cy"],
                                               _bbox_metrics(item[0])["cx"]))
    text = "".join(item[1].strip() for item in ordered)
    norm = normalize_text(text)
    weight = sum(max(1, len(item[3])) for item in ordered)
    confidence = sum(item[2] * max(1, len(item[3])) for item in ordered) / max(1, weight)
    return (_enclosing_bbox([item[0] for item in ordered]), text, confidence, norm, "vertical")


def _vertical_columns_belong_together(first, second):
    """Whether two vertical columns are likely one multi-column Japanese block."""
    one, two = _bbox_metrics(first[0]), _bbox_metrics(second[0])
    if _vertical_overlap(one, two) < 0.30:
        return False

    height_ratio = max(one["height"], two["height"]) / max(1.0, min(one["height"], two["height"]))
    if height_ratio > 2.8:
        return False

    left, right = (one, two) if one["cx"] <= two["cx"] else (two, one)
    horizontal_gap = right["left"] - left["right"]
    max_width = max(one["width"], two["width"])
    center_dx = abs(one["cx"] - two["cx"])

    # Vertical Japanese columns are usually spaced by roughly one character
    # width. Allow a little extra for anime title cards / credits, but avoid
    # combining unrelated signs that merely share a y-range.
    return (horizontal_gap <= max(24.0, max_width * 1.65) and
            center_dx <= max(36.0, max_width * 3.0))


def _merge_vertical_columns(columns):
    merged = []
    for group in _union_groups(columns, _vertical_columns_belong_together):
        # Traditional tategaki is read from the rightmost column to the left.
        # Within each column the text has already been assembled top-to-bottom.
        ordered = sorted(group, key=lambda item: _bbox_metrics(item[0])["cx"], reverse=True)
        text = "\n".join(item[1] for item in ordered)
        norm = normalize_text(text)
        weight = sum(max(1, len(item[3])) for item in ordered)
        confidence = sum(item[2] * max(1, len(item[3])) for item in ordered) / max(1, weight)
        merged.append((_enclosing_bbox([item[0] for item in ordered]), text,
                       confidence, norm, "vertical"))
    return merged


def _prepare_detections(raw_detections):
    """Normalize one OCR frame, including reconstruction of Japanese tategaki.

    EasyOCR often emits vertical Japanese as one- or two-character fragments.
    We retain those short Japanese fragments long enough to assemble top-to-bottom
    columns, then combine adjacent columns in the Japanese right-to-left reading
    order. Horizontal detections continue to use the previous filtering rules.
    """
    candidates = []
    for detection in raw_detections:
        if not isinstance(detection, (list, tuple)) or len(detection) < 3:
            continue
        bbox, text, confidence = detection[:3]
        norm = normalize_text(text)
        if float(confidence) < 0.30 or not norm or not contains_japanese(text):
            continue
        native_bbox = [[float(point[0]), float(point[1])] for point in bbox]
        candidates.append((native_bbox, text, float(confidence), norm))

    # Prefer the largest/best rendering of a nested duplicate, but do this before
    # vertical grouping so duplicate character boxes do not create fake columns.
    candidates.sort(key=lambda item: (len(item[3]), item[2]), reverse=True)
    accepted = []
    for candidate in candidates:
        bbox, _, _, norm = candidate
        nested = any(_spatially_nested(bbox, kept[0]) and (norm in kept[3] or kept[3] in norm)
                     for kept in accepted)
        if not nested:
            accepted.append(candidate)

    vertical_members = set()
    vertical_columns = []
    for group in _union_groups(accepted, _vertical_fragment_pair):
        if not _looks_like_vertical_column(group):
            continue
        vertical_columns.append(_make_vertical_column(group))
        vertical_members.update(id(item) for item in group)

    vertical_blocks = _merge_vertical_columns(vertical_columns) if vertical_columns else []

    horizontal = []
    for item in accepted:
        if id(item) in vertical_members:
            continue
        bbox, text, confidence, norm = item
        # Preserve the old behavior for standalone horizontal detections: very
        # short results are too noisy to surface as independent signs.
        if len(norm) < 2:
            continue
        horizontal.append((bbox, text, confidence, norm, "horizontal"))

    results = horizontal + vertical_blocks
    results.sort(key=lambda item: (_bbox_metrics(item[0])["top"], _bbox_metrics(item[0])["left"]))
    return results


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


def _read_detections(reader, image):
    """Run EasyOCR with detail needed for vertical reconstruction."""
    return _prepare_detections(reader.readtext(image, detail=1, paragraph=False))


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
    detected = _read_detections(reader, crop)
    return any(ratio(norm_target, norm) >= 78 or norm_target in norm or norm in norm_target
               for _, _, _, norm, _ in detected)


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
    cache_name = f"ocr_signs_{_OCR_CACHE_VERSION}"
    cached = cache.load_json(video_file, cache_name)
    if cached is not None:
        resolution = _resolution(video_file)
        # Enrich records missing a group centroid so ASS can use exact coordinates.
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
            for bbox, text, confidence, norm, orientation in _read_detections(reader, frame):
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
                                   "bbox": bbox, "pos": pos, "x": x, "y": y,
                                   "orientation": orientation})
                    found = len(active) - 1
                else:
                    if len(norm) > len(active[found]["norm_text"]):
                        active[found].update({"ja_text": text, "norm_text": norm, "bbox": bbox,
                                              "x": detected_x, "y": detected_y,
                                              "orientation": orientation})
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
    cache.save_json(video_file, cache_name, finished)
    return finished, (width, height)


def translate_signs(signs, video_file: str, manager: GeminiManager, cache: CacheStore,
                    executor: concurrent.futures.Executor) -> List[Subtitle]:
    signs = [sign for sign in signs if contains_japanese(sign.get("ja_text", ""))]

    def translate(index, batch):
        name = f"gemini_ocr_{_OCR_CACHE_VERSION}_batch_{index}"
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
