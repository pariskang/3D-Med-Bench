"""Sign Rendering Library — maps clinical sign IDs to Godot/UE5 render params."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from clinicraft.config import settings


class RenderParams(BaseModel):
    sign_id: str
    blendshapes: dict[str, float] = {}
    animation: str | None = None
    skin_color_rgb: list[int] | None = None
    texture_overrides: dict[str, str] = {}
    posture: str | None = None
    audio_clip: str | None = None
    description: str = ""
    tier: str = "T1"


class SignRenderLibrary:
    def __init__(self, entries: dict[str, dict]) -> None:
        self._entries = entries

    @classmethod
    def load(cls, path: Path | None = None) -> "SignRenderLibrary":
        path = path or settings.sign_lib_path
        if not path.exists():
            return cls({})
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def resolve(
        self, sign_id: str, severity: str = "moderate"
    ) -> RenderParams | None:
        entry = self._entries.get(sign_id)
        if not entry:
            return None
        sev_data = entry.get("severity_overrides", {}).get(severity, {})
        merged = {**entry.get("base", {}), **sev_data, "sign_id": sign_id}
        return RenderParams.model_validate(merged)

    def list_signs(self) -> list[str]:
        return list(self._entries.keys())
