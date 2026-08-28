import json
import re


_PUNCTUATION = re.compile(r"[\s\u3000、。！？「」『』（）,.?!♪~～]")


def normalize_text(text: str) -> str:
    return _PUNCTUATION.sub("", text or "").casefold()


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
        return []
