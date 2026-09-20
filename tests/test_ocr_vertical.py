from anime_subber_core.ocr import _prepare_detections


def _box(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def test_vertical_fragments_are_read_top_to_bottom():
    detections = [
        (_box(100, 80, 130, 110), "語", 0.94),
        (_box(100, 10, 130, 40), "日", 0.96),
        (_box(100, 45, 130, 75), "本", 0.95),
    ]

    prepared = _prepare_detections(detections)

    assert len(prepared) == 1
    assert prepared[0][1] == "日本語"
    assert prepared[0][4] == "vertical"


def test_vertical_columns_are_read_right_to_left():
    detections = [
        (_box(60, 45, 90, 75), "文", 0.91),
        (_box(110, 45, 140, 75), "本", 0.95),
        (_box(60, 10, 90, 40), "語", 0.92),
        (_box(110, 10, 140, 40), "日", 0.96),
    ]

    prepared = _prepare_detections(detections)

    assert len(prepared) == 1
    assert prepared[0][1] == "日本\n語文"
    assert prepared[0][3] == "日本語文"
    assert prepared[0][4] == "vertical"


def test_distant_vertical_columns_are_not_merged():
    detections = [
        (_box(20, 10, 50, 40), "日", 0.95),
        (_box(20, 45, 50, 75), "本", 0.95),
        (_box(300, 10, 330, 40), "語", 0.95),
        (_box(300, 45, 330, 75), "文", 0.95),
    ]

    prepared = _prepare_detections(detections)

    assert [item[1] for item in prepared] == ["日本", "語文"]
    assert all(item[4] == "vertical" for item in prepared)


def test_horizontal_japanese_is_preserved():
    detections = [(_box(20, 200, 180, 235), "こんにちは", 0.98)]

    prepared = _prepare_detections(detections)

    assert len(prepared) == 1
    assert prepared[0][1] == "こんにちは"
    assert prepared[0][4] == "horizontal"
