from dataclasses import dataclass
from typing import Optional


@dataclass
class Subtitle:
    start: float
    end: float
    text: str
    position: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    layer: int = 0
    font_size: Optional[float] = None

    def clamp(self, lower: float, upper: float) -> "Subtitle":
        self.start = min(max(lower, self.start), upper)
        self.end = min(max(self.start + 0.05, self.end), upper)
        return self
