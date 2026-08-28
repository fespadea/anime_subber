import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


class CacheStore:
    """Stores every intermediate below the project-local .cache directory."""

    def __init__(self, root: Optional[Path] = None):
        project_root = Path(__file__).resolve().parent.parent
        self.root = Path(root) if root else project_root / ".cache" / "anime_subber"
        self.root.mkdir(parents=True, exist_ok=True)

    def media_dir(self, media_path: str) -> Path:
        resolved = str(Path(media_path).resolve())
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(media_path).stem)[:80]
        path = self.root / f"{safe_stem}-{digest}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, media_path: str, name: str, suffix: str = ".json") -> Path:
        return self.media_dir(media_path) / f"{name}{suffix}"

    def load_json(self, media_path: str, name: str) -> Any:
        path = self.path(media_path, name)
        if not path.exists():
            # One-time, read-compatible migration from the original layout.
            legacy = Path(str(Path(media_path).with_suffix("")) + f".{name}.json")
            if legacy.exists():
                try:
                    with legacy.open("r", encoding="utf-8") as stream:
                        value = json.load(stream)
                    self.save_json(media_path, name, value)
                    print(f"[Cache] Migrated {legacy.name} into {path.parent}")
                    return value
                except (OSError, json.JSONDecodeError):
                    pass
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None

    def save_json(self, media_path: str, name: str, value: Any) -> Path:
        destination = self.path(media_path, name)
        fd, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination
