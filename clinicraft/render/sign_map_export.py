"""
Export the Sign Rendering Library (signs.yaml) → a flat JSON map the Godot
renderer reads at runtime (res://data/sign_render_map.json).

Keeps the GDScript avatar and the Python-side Sign Rendering Library in sync
from a single source of truth. Fully testable in Python (no Godot needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from clinicraft.config import settings


def build_sign_map(sign_lib_path: Path | None = None) -> dict[str, Any]:
    """Flatten signs.yaml into {sign_id: {render params}} for the renderer."""
    sign_lib_path = sign_lib_path or settings.sign_lib_path
    data = yaml.safe_load(sign_lib_path.read_text(encoding="utf-8")) or {}

    out: dict[str, Any] = {}
    for sign_id, entry in data.items():
        base = entry.get("base", {})
        rec: dict[str, Any] = {
            "description": entry.get("description", ""),
            "tier": entry.get("tier", "T1"),
        }
        if "skin_color_rgb" in base:
            rec["skin_color_rgb"] = base["skin_color_rgb"]
        if "blendshapes" in base:
            rec["blendshapes"] = base["blendshapes"]
        if "animation" in base:
            rec["animation"] = base["animation"]
        if "posture" in base:
            rec["posture"] = base["posture"]
        if "texture_overrides" in base:
            rec["texture_overrides"] = base["texture_overrides"]
        if "audio_clip" in base:
            rec["audio_clip"] = base["audio_clip"]
        # severity variants (renderer may pick one)
        if "severity_overrides" in entry:
            rec["severity_overrides"] = entry["severity_overrides"]
        out[sign_id] = rec
    return out


def export_sign_map(
    sign_lib_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write the sign map JSON into the Godot project's data directory."""
    out_path = out_path or (
        settings.resources_dir / "avatars_cc0" / "godot_project"
        / "data" / "sign_render_map.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sign_map = build_sign_map(sign_lib_path)
    out_path.write_text(
        json.dumps(sign_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path
