"""Perception manager for vision-detected blocks and coordinate mapping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Default table surface height in mm (used when Z is not provided by camera)
_DEFAULT_TABLE_Z: float = 20.0

# Counter for auto-generating IDs when camera doesn't provide them
_auto_id_counter: dict[str, int] = {}


@dataclass
class BlockObject:
    """Represents a physical object detected in the robot's workspace."""

    id: str
    color: str
    type: str
    x: float
    y: float
    z: float
    r: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockObject:
        """Create a BlockObject from a dict.

        Minimal required fields: color, x, y
        Optional fields: id, type, z, r (auto-filled with defaults)
        """
        color = str(data.get("color", "unknown")).lower()
        obj_type = str(data.get("type", "block")).lower()

        # Auto-generate ID if not provided
        obj_id = data.get("id", "")
        if not obj_id:
            key = f"{color}_{obj_type}"
            _auto_id_counter[key] = _auto_id_counter.get(key, 0) + 1
            obj_id = f"{color}_{obj_type}_{_auto_id_counter[key]}"

        return cls(
            id=str(obj_id),
            color=color,
            type=obj_type,
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data.get("z", _DEFAULT_TABLE_Z)),
            r=float(data.get("r", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



class PerceptionManager:
    """Manages perceived objects and provides query and LLM prompt context."""

    def __init__(self, blocks: list[BlockObject] | None = None, source_path: Path | None = None) -> None:
        self.blocks: list[BlockObject] = blocks or []
        self._source_path: Path | None = source_path

    @classmethod
    def from_file(cls, path: str | Path) -> PerceptionManager:
        file_path = Path(path)
        if not file_path.exists():
            return cls([], source_path=file_path)
        try:
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)
            raw_blocks = data.get("blocks", data) if isinstance(data, dict) else data
            blocks = [BlockObject.from_dict(b) for b in raw_blocks if isinstance(b, dict)]
            return cls(blocks, source_path=file_path)
        except Exception:
            return cls([], source_path=file_path)

    def reload(self, path: str | Path | None = None) -> None:
        """Reload blocks dynamically from disk so live camera updates are immediately visible."""
        if path is not None:
            self._source_path = Path(path)
        if self._source_path and self._source_path.exists():
            try:
                content = self._source_path.read_text(encoding="utf-8")
                data = json.loads(content)
                raw_blocks = data.get("blocks", data) if isinstance(data, dict) else data
                self.blocks = [BlockObject.from_dict(b) for b in raw_blocks if isinstance(b, dict)]
            except Exception:
                pass

    def save_to_file(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"blocks": [b.to_dict() for b in self.blocks]}
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_block(self, block: BlockObject) -> None:
        self.blocks = [b for b in self.blocks if b.id != block.id]
        self.blocks.append(block)

    def find(
        self,
        *,
        color: str | None = None,
        type: str | None = None,
        name: str | None = None,
        id: str | None = None,
    ) -> BlockObject | None:
        """Find the best-matching block based on specified attributes."""
        target_color = color.lower().strip() if color else None
        target_type = type.lower().strip() if type else None
        target_name = name.lower().strip() if name else None
        target_id = id.lower().strip() if id else None

        for b in self.blocks:
            if target_id and b.id.lower() == target_id:
                return b
            if target_color and b.color == target_color:
                if target_type and b.type != target_type:
                    continue
                return b
            if target_name and (b.id.lower() == target_name or b.color in target_name):
                return b
            if target_type and b.type == target_type and not target_color:
                return b
        return None

    def to_prompt_context(self) -> str:
        """Formats perceived objects into context text for the LLM prompt."""
        if not self.blocks:
            return "No objects currently detected in the workspace."

        lines = ["Detected objects currently visible on the work table:"]
        for b in self.blocks:
            lines.append(
                f"- {b.color} {b.type} (id: {b.id}) at X={b.x:.1f} mm, Y={b.y:.1f} mm, Z={b.z:.1f} mm"
            )
        return "\n".join(lines)
