"""Japanese phonetic normalization used as a secondary alignment signal."""
import functools
import re


@functools.lru_cache(maxsize=1)
def _converter():
    try:
        from pykakasi import kakasi
    except ImportError:
        return None
    return kakasi()


@functools.lru_cache(maxsize=4096)
def phonetic_text(text: str) -> str:
    """Return a loose hiragana reading, or an empty string without pykakasi."""
    converter = _converter()
    if converter is None:
        return ""
    reading = converter.convert(text or "")
    value = "".join(item.get("hira", item.get("orig", "")) for item in reading)
    value = "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in value)
    value = re.sub(r"[\s\u3000、。！？「」『』（）,.?!♪~～]", "", value).casefold()
    return value.replace("ー", "").replace("っ", "")
