"""On-screen Japanese text detection and translation (kept separate from audio alignment)."""
import concurrent.futures
import hashlib
import re
from typing import List, Tuple

from .cache import CacheStore
from .gemini import (GeminiManager, recognize_japanese_image_batch,
                     translate_text_batch)
from .models import Subtitle
from .similarity import ratio
from .text import normalize_text


_KANJI = re.compile(r"[一-龯々〆ヵヶ]")
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_LATIN = re.compile(r"[A-Za-z]")


# OCR cache versions are intentionally tied to the detection/layout algorithm.
# v7 adds permissive text detection, geometry-first vertical rescue, Gemini
# vision verification for tategaki crops, and more stable spatial tracking.
_OCR_CACHE_VERSION = "v7"


_EASYOCR_READTEXT_KWARGS = {
    # CRAFT's defaults are conservative for thin handwritten anime text. Keep
    # recognition filtering strict downstream, but ask the detector to retain
    # weaker candidate boxes so vertical geometry can rescue them.
    "detail": 1,
    "paragraph": False,
    "text_threshold": 0.50,
    "low_text": 0.25,
    "link_threshold": 0.25,
    "canvas_size": 3840,
    "mag_ratio": 1.50,
    "min_size": 8,
    "add_margin": 0.12,
}


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


def _bbox_area(bbox):
    metrics = _bbox_metrics(bbox)
    return metrics["width"] * metrics["height"]


def _bbox_iou(first, second):
    a = _bbox_metrics(first)
    b = _bbox_metrics(second)
    left, top = max(a["left"], b["left"]), max(a["top"], b["top"])
    right, bottom = min(a["right"], b["right"]), min(a["bottom"], b["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / max(1.0, union)


def _same_region(first, second, width, height, loose=False):
    """Spatial sign identity independent of OCR wording.

    OCR text is unstable for stylized signs, especially tategaki. A nested box
    or strong IoU is therefore stronger tracking evidence than exact text.
    ``loose`` is reserved for geometry-only vertical rescue candidates.
    """
    one, two = _bbox_metrics(first), _bbox_metrics(second)
    x_limit = max(70.0, width * (0.10 if loose else 0.07))
    y_limit = max(55.0, height * (0.10 if loose else 0.07))
    if abs(one["cx"] - two["cx"]) > x_limit or abs(one["cy"] - two["cy"]) > y_limit:
        return False
    if _bbox_iou(first, second) >= (0.32 if loose else 0.55):
        return True
    return _spatially_nested(first, second, threshold=0.52 if loose else 0.68)


def _coerce_detection(detection):
    if not isinstance(detection, (list, tuple)) or len(detection) < 3:
        return None
    bbox, text, confidence = detection[:3]
    try:
        native_bbox = [[float(point[0]), float(point[1])] for point in bbox]
        confidence = float(confidence)
    except (TypeError, ValueError, IndexError):
        return None
    if len(native_bbox) < 4:
        return None
    text = str(text or "")
    return native_bbox, text, confidence, normalize_text(text)


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
        candidate = _coerce_detection(detection)
        if candidate is None:
            continue
        bbox, text, confidence, norm = candidate
        if confidence < 0.30 or not norm or not contains_japanese(text):
            continue
        candidates.append((bbox, text, confidence, norm))

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


def _looks_like_vertical_region(group):
    """Geometry-only vertical candidate test for weak/garbled OCR results."""
    if not group:
        return False
    box = _enclosing_bbox([item[0] for item in group])
    metrics = _bbox_metrics(box)
    if metrics["height"] < 24 or metrics["width"] < 4:
        return False
    if len(group) == 1:
        # A detector may give one tall box even when recognition fails.
        return metrics["height"] >= metrics["width"] * 1.45
    # Multiple stacked boxes are a stronger vertical signal, so a slightly
    # lower aspect ratio is acceptable for handwritten/multi-glyph regions.
    return metrics["height"] >= metrics["width"] * 1.10


def _merge_vertical_region_boxes(columns):
    merged = []
    for group in _union_groups(columns, _vertical_columns_belong_together):
        merged.append(_enclosing_bbox([item[0] for item in group]))
    return merged


def _vertical_rescue_regions(raw_detections, prepared, width, height):
    """Find likely tategaki regions even when recognition itself is unusable.

    EasyOCR's detector can often localize thin vertical text while its recognizer
    returns low-confidence Latin/punctuation garbage. Previous code discarded
    those boxes before vertical reconstruction, which made recovery impossible.
    This pass uses box geometry first and leaves recognition to Gemini vision.
    """
    raw = []
    for detection in raw_detections:
        candidate = _coerce_detection(detection)
        if candidate is None:
            continue
        bbox, text, confidence, norm = candidate
        metrics = _bbox_metrics(bbox)
        # Keep very weak recognitions, but reject microscopic and implausibly
        # huge regions that are typically detector noise/artwork edges.
        if confidence < 0.01:
            continue
        if metrics["width"] < 4 or metrics["height"] < 6:
            continue
        if metrics["width"] > width * 0.70 or metrics["height"] > height * 0.92:
            continue
        raw.append((bbox, text, confidence, norm))

    columns = []
    for group in _union_groups(raw, _vertical_fragment_pair):
        if not _looks_like_vertical_region(group):
            continue
        confidence = max((item[2] for item in group), default=0.0)
        columns.append((_enclosing_bbox([item[0] for item in group]), "", confidence, "", "vertical"))

    regions = _merge_vertical_region_boxes(columns) if columns else []
    prepared_vertical = [item[0] for item in prepared if item[4] == "vertical"]
    result = []
    for bbox in regions:
        # A successfully reconstructed vertical detection already has Japanese
        # text. It will still be vision-verified later; don't emit a duplicate
        # geometry-only candidate for the same box.
        if any(_same_region(bbox, existing, width, height, loose=True)
               for existing in prepared_vertical):
            continue
        metrics = _bbox_metrics(bbox)
        # Ignore regions that are technically tall but only by a few pixels.
        if metrics["height"] < max(32.0, height * 0.025):
            continue
        result.append(bbox)
    return result


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
    """Run a recall-oriented EasyOCR pass, then apply strict text filtering."""
    raw = reader.readtext(image, **_EASYOCR_READTEXT_KWARGS)
    return _prepare_detections(raw)


def _read_frame_ocr(reader, image):
    """Return recognized signs plus geometry-only vertical rescue regions."""
    height, width = image.shape[:2]
    raw = reader.readtext(image, **_EASYOCR_READTEXT_KWARGS)
    prepared = _prepare_detections(raw)
    rescue = _vertical_rescue_regions(raw, prepared, width, height)
    return prepared, rescue


def _region_matches_target(candidate_bbox, target_bbox, crop_width, crop_height):
    # Coordinates are local to the same crop. Be somewhat permissive because
    # EasyOCR can resize/merge the same vertical line differently frame-to-frame.
    return _same_region(candidate_bbox, target_bbox, crop_width, crop_height, loose=True)


def _frame_has_sign(cap, reader, frame_index, norm_target, bbox, spatial_only=False):
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
    if spatial_only:
        detected, rescue = _read_frame_ocr(reader, crop)
        # Translate the target box into crop-local coordinates.
        local_target = [[point[0] - x1, point[1] - y1] for point in bbox]
        boxes = [item[0] for item in detected if item[4] == "vertical"] + rescue
        return any(_region_matches_target(candidate, local_target, crop.shape[1], crop.shape[0])
                   for candidate in boxes)

    detected = _read_detections(reader, crop)
    return any(ratio(norm_target, norm) >= 78 or norm_target in norm or norm in norm_target
               for _, _, _, norm, _ in detected)


def _first_visible_frame(cap, reader, hit_frame, stride, norm_target, bbox, spatial_only=False):
    low, high, first = max(0, hit_frame - stride + 1), hit_frame, hit_frame
    while low <= high:
        middle = (low + high) // 2
        visible = (_frame_has_sign(cap, reader, middle, norm_target, bbox, spatial_only=True)
                   if spatial_only else
                   _frame_has_sign(cap, reader, middle, norm_target, bbox))
        if visible:
            first, high = middle, middle - 1
        else:
            low = middle + 1
    return first


def _last_visible_frame(cap, reader, last_hit, first_miss, norm_target, bbox, spatial_only=False):
    low, high, last = last_hit, max(last_hit, first_miss - 1), last_hit
    while low <= high:
        middle = (low + high) // 2
        visible = (_frame_has_sign(cap, reader, middle, norm_target, bbox, spatial_only=True)
                   if spatial_only else
                   _frame_has_sign(cap, reader, middle, norm_target, bbox))
        if visible:
            last, low = middle, middle + 1
        else:
            high = middle - 1
    return last


def _text_track_match(norm, existing_norm):
    if not norm or not existing_norm:
        return False
    return (ratio(norm, existing_norm) >= 82 or norm in existing_norm or existing_norm in norm)


def _track_match(sign, bbox, norm, orientation, width, height):
    """Match a frame detection to an active sign using space first, text second."""
    rescue = orientation == "vertical_rescue" or sign.get("needs_vision", False)
    if not _same_region(bbox, sign["bbox"], width, height, loose=rescue):
        return False
    if rescue:
        return True
    if _text_track_match(norm, sign.get("norm_text", "")):
        return True
    # Very stable geometry can bridge moderate OCR wording changes without
    # merging unrelated signs that merely occupy the same screen quadrant.
    return _bbox_iou(bbox, sign["bbox"]) >= 0.72 and ratio(norm, sign.get("norm_text", "")) >= 35


def _sample_score(bbox, confidence, norm):
    # Prefer a complete/larger rendering and, secondarily, stronger recognition.
    return _bbox_area(bbox) * (0.25 + max(0.0, confidence)) * (1.0 + min(12, len(norm)) / 12.0)


def _crop_frame_jpeg(video_file, frame_index, bbox):
    """Extract a generously padded representative crop for Gemini vision."""
    import cv2

    cap = cv2.VideoCapture(video_file)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        if not ok:
            return None
    finally:
        cap.release()

    frame_height, frame_width = frame.shape[:2]
    metrics = _bbox_metrics(bbox)
    pad_x = max(24, round(metrics["width"] * 0.28))
    pad_y = max(18, round(metrics["height"] * 0.08))
    x1 = max(0, round(metrics["left"]) - pad_x)
    x2 = min(frame_width, round(metrics["right"]) + pad_x)
    y1 = max(0, round(metrics["top"]) - pad_y)
    y2 = min(frame_height, round(metrics["bottom"]) + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return encoded.tobytes() if ok else None


def _resolve_vertical_vision(signs, video_file, manager, executor, batch_size=12):
    """Verify/recover vertical Japanese using Gemini on representative crops."""
    pending = []
    for sign in signs:
        if not sign.get("needs_vision"):
            continue
        image = _crop_frame_jpeg(video_file, sign.get("sample_frame", sign.get("last_seen_frame", 0)),
                                 sign["bbox"])
        if image:
            pending.append((sign, image))

    def recognize(batch):
        items = [{"id": index, "image_bytes": image}
                 for index, (_sign, image) in enumerate(batch)]
        data = recognize_japanese_image_batch(items, manager)
        mapping = {int(item["id"]): item["ja"] for item in data}
        return [(sign, mapping.get(index, "")) for index, (sign, _image) in enumerate(batch)]

    futures = []
    for offset in range(0, len(pending), batch_size):
        futures.append(executor.submit(recognize, pending[offset:offset + batch_size]))

    for future in futures:
        for sign, japanese in future.result():
            japanese = str(japanese or "").strip()
            if japanese and contains_japanese(japanese):
                sign["ja_text"] = japanese
                sign["norm_text"] = normalize_text(japanese)
                sign["orientation"] = "vertical"
                sign["vision_verified"] = True

    # Geometry-only false positives have no source text to translate. If Gemini
    # was unavailable, keep vertical signs that EasyOCR had already recognized.
    return [sign for sign in signs if sign.get("norm_text")]


def _sign_fingerprint(sign):
    """Stable-enough provenance token for future ASS OCR refreshes."""
    metrics = _bbox_metrics(sign.get("bbox") or [[0, 0], [0, 0], [0, 0], [0, 0]])
    payload = "|".join((
        normalize_text(sign.get("ja_text", "")),
        str(round(metrics["cx"] / 16)),
        str(round(metrics["cy"] / 16)),
        str(round(metrics["width"] / 16)),
        str(round(metrics["height"] / 16)),
    ))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def detect_signs(video_file: str, cache: CacheStore, gpu=False, sample_seconds=1.0,
                 manager=None, executor=None, vision_rescue=True):
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

    def finish(sign, first_miss):
        spatial_only = bool(sign.get("needs_vision"))
        last_frame = _last_visible_frame(
            cap, reader, sign["last_seen_frame"], first_miss, sign.get("norm_text", ""),
            sign["bbox"], spatial_only=spatial_only,
        )
        sign["end"] = min(total / fps, (last_frame + 1) / fps)
        finished.append(sign)

    try:
        for frame_index in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            now = frame_index / fps
            seen = set()

            if vision_rescue:
                recognized, rescue_boxes = _read_frame_ocr(reader, frame)
                frame_detections = list(recognized) + [
                    (bbox, "", 0.0, "", "vertical_rescue") for bbox in rescue_boxes
                ]
            else:
                frame_detections = _read_detections(reader, frame)

            for bbox, text, confidence, norm, orientation in frame_detections:
                needs_vision = vision_rescue and orientation in ("vertical", "vertical_rescue")
                _, detected_x, detected_y = _position(bbox, width, height)
                found = next((
                    index for index, sign in enumerate(active)
                    if _track_match(sign, bbox, norm, orientation, width, height)
                ), None)

                score = _sample_score(bbox, confidence, norm)
                if found is None:
                    pos, x, y = _position(bbox, width, height)
                    first_frame = _first_visible_frame(
                        cap, reader, frame_index, stride, norm, bbox, spatial_only=needs_vision
                    )
                    active.append({
                        "start": first_frame / fps,
                        "last_seen": now,
                        "last_seen_frame": frame_index,
                        "ja_text": text,
                        "norm_text": norm,
                        "bbox": bbox,
                        "pos": pos,
                        "x": x,
                        "y": y,
                        "orientation": "vertical" if needs_vision else orientation,
                        "confidence": confidence,
                        "needs_vision": needs_vision,
                        "sample_frame": frame_index,
                        "sample_score": score,
                    })
                    found = len(active) - 1
                else:
                    sign = active[found]
                    if needs_vision:
                        sign["needs_vision"] = True
                        sign["orientation"] = "vertical"
                    # Retain the strongest/most complete source transcription,
                    # but track presence primarily from geometry.
                    if norm and (
                        len(norm) > len(sign.get("norm_text", "")) or
                        confidence > sign.get("confidence", 0.0) + 0.12
                    ):
                        sign["ja_text"] = text
                        sign["norm_text"] = norm
                        sign["confidence"] = confidence
                    if score > sign.get("sample_score", -1):
                        pos, x, y = _position(bbox, width, height)
                        sign.update({
                            "bbox": bbox, "pos": pos, "x": x, "y": y,
                            "sample_frame": frame_index, "sample_score": score,
                        })
                    sign["last_seen"] = now
                    sign["last_seen_frame"] = frame_index
                seen.add(found)

            for index in reversed(range(len(active))):
                if index not in seen and now - active[index]["last_seen"] >= sample_seconds:
                    finish(active.pop(index), frame_index)
    finally:
        cap.release()

    duration = total / fps
    if active:
        boundary_cap = cv2.VideoCapture(video_file)
        try:
            # ``finish`` closes over ``cap`` which has now been released, so do
            # the final boundary refinement against a fresh capture explicitly.
            for sign in active:
                last_frame = _last_visible_frame(
                    boundary_cap, reader, sign["last_seen_frame"], total,
                    sign.get("norm_text", ""), sign["bbox"],
                    spatial_only=bool(sign.get("needs_vision")),
                )
                sign["end"] = min(duration, (last_frame + 1) / fps)
                finished.append(sign)
        finally:
            boundary_cap.release()

    if vision_rescue and manager is not None and executor is not None:
        finished = _resolve_vertical_vision(finished, video_file, manager, executor)
    else:
        # Pure geometry candidates cannot be translated without the vision pass.
        finished = [sign for sign in finished if sign.get("norm_text")]

    # A final containment pass catches a component that was intermittently
    # recognized as a separate sign on different sampled frames.
    deduplicated = []
    for sign in sorted(finished, key=lambda item: (-len(item.get("norm_text", "")), item["start"])):
        parent = next((
            kept for kept in deduplicated
            if min(sign["end"], kept["end"]) - max(sign["start"], kept["start"]) > 0.05
            and _same_region(sign["bbox"], kept["bbox"], width, height,
                             loose=(sign.get("orientation") == "vertical" or
                                    kept.get("orientation") == "vertical"))
            and (_text_track_match(sign.get("norm_text", ""), kept.get("norm_text", ""))
                 or ratio(sign.get("norm_text", ""), kept.get("norm_text", "")) >= 55)
        ), None)
        if parent:
            parent["start"] = min(parent["start"], sign["start"])
            parent["end"] = max(parent["end"], sign["end"])
        else:
            deduplicated.append(sign)

    # Internal tracking fields do not belong in the persistent OCR cache.
    for sign in deduplicated:
        for key in ("confidence", "needs_vision", "sample_frame", "sample_score"):
            sign.pop(key, None)

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
        mapping = {}
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                identifier = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            english = str(item.get("en", "")).strip()
            if english:
                mapping[identifier] = english
        return [Subtitle(float(sign["start"]), float(sign["end"]), f"[{mapping[i]}]",
                         int(sign["pos"]), float(sign["x"]), float(sign["y"]), 1,
                         effect=f"anime_subber_ocr:{_sign_fingerprint(sign)}")
                for i, sign in enumerate(batch) if mapping.get(i)]

    batches = [signs[i:i + 50] for i in range(0, len(signs), 50)]
    futures = [executor.submit(translate, i, batch) for i, batch in enumerate(batches)]
    return [cue for future in futures for cue in future.result()]


def process_video_signs(video_file, manager, cache, executor, gpu=False, vision_rescue=True):
    signs, resolution = detect_signs(video_file, cache, gpu, manager=manager, executor=executor,
                                     vision_rescue=vision_rescue)
    return translate_signs(signs, video_file, manager, cache, executor), resolution
