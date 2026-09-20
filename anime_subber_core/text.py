import json
import unicodedata


_IGNORED_SYMBOLS = frozenset("♪♫♬~〜～")


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching without erasing Japanese letters.

    NFKC folds full-width/half-width variants, while Unicode punctuation and
    whitespace are ignored so Gemini/Whisper punctuation choices do not affect
    alignment. A few common music/decorative symbols are ignored explicitly.
    """
    value = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(
        character for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
        and character not in _IGNORED_SYMBOLS
    )


def parse_llm_json(text: str):
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(cleaned)
        return value
    except (json.JSONDecodeError, TypeError):
        return None
