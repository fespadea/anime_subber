"""On-screen Japanese text detection and translation (kept separate from audio alignment)."""
import concurrent.futures
from typing import List, Tuple

from .cache import CacheStore
from .gemini import GeminiManager, translate_text_batch
from .models import Subtitle
from .similarity import ratio
from .text import normalize_text


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


def detect_signs(video_file: str, cache: CacheStore, gpu=False, sample_seconds=1.0):
    cached = cache.load_json(video_file, "ocr_signs")
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
            for bbox, text, confidence in reader.readtext(frame):
                norm = normalize_text(text)
                if confidence < 0.35 or len(norm) < 2:
                    continue
                found = next((i for i, sign in enumerate(active) if ratio(norm, sign["norm_text"]) >= 82), None)
                if found is None:
                    pos, x, y = _position(bbox, width, height)
                    active.append({"start": now, "last_seen": now, "ja_text": text, "norm_text": norm,
                                   "bbox": bbox, "pos": pos, "x": x, "y": y})
                    found = len(active) - 1
                else:
                    active[found]["last_seen"] = now
                seen.add(found)
            for index in reversed(range(len(active))):
                if index not in seen and now - active[index]["last_seen"] >= sample_seconds:
                    sign = active.pop(index)
                    sign["end"] = min(now, sign["last_seen"] + sample_seconds)
                    finished.append(sign)
    finally:
        cap.release()
    duration = total / fps
    for sign in active:
        sign["end"] = min(duration, sign["last_seen"] + sample_seconds)
        finished.append(sign)
    cache.save_json(video_file, "ocr_signs", finished)
    return finished, (width, height)


def translate_signs(signs, video_file: str, manager: GeminiManager, cache: CacheStore,
                    executor: concurrent.futures.Executor) -> List[Subtitle]:
    def translate(index, batch):
        name = f"gemini_ocr_batch_{index}"
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
