"""Small fuzzy matching facade with a standard-library test fallback."""
from difflib import SequenceMatcher

try:
    from thefuzz import fuzz as _fuzz
except ImportError:  # Unit tests and --help do not require optional runtime packages.
    _fuzz = None


def ratio(left: str, right: str) -> int:
    if _fuzz:
        return _fuzz.ratio(left, right)
    return round(100 * SequenceMatcher(None, left, right).ratio())


def partial_ratio(left: str, right: str) -> int:
    if _fuzz:
        return _fuzz.partial_ratio(left, right)
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return 100
    matcher = SequenceMatcher(None, left, right)
    return round(100 * max((SequenceMatcher(None, left, right[max(0, block.b - block.a):][:len(left)]).ratio()
                            for block in matcher.get_matching_blocks()), default=0))
