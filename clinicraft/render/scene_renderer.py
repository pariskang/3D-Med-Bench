"""
Scene renderer abstraction (Tier A).

Two implementations:
  GodotSceneRenderer — drives a Godot 4 headless process to render the avatar
      with injected sign blendshapes/animations, returns PNG frame paths.
      Requires settings.godot_bin to point at a Godot 4 binary with the
      CliniCraft avatar project. NOT bundled — this is the integration seam.
  StubSceneRenderer — no Godot available. Produces a single labelled placeholder
      PNG per turn (via Pillow) that visually encodes which signs are present,
      so the frame_stream path is end-to-end runnable and the multimodal judge
      has a real image to score. This is explicitly NOT photoreal and is logged
      as such — it exists so the loop and scoring are testable before the real
      renderer lands.

Honest status: the photoreal Godot/UE5 pipeline is future work (§11 P1). The
stub keeps the benchmark runnable and lets C3 scoring exercise the multimodal
path, but stub frames must not be used for published 3D-perception results.
"""

from __future__ import annotations

import base64
import hashlib
import subprocess
from pathlib import Path
from typing import Protocol

from loguru import logger

from clinicraft.config import settings
from clinicraft.render.sign_library import SignRenderLibrary
from clinicraft.schemas.case_pack import PatientConfig
from clinicraft.schemas.ground_truth import VisibleSign


class SceneRenderer(Protocol):
    def render(
        self, signs: list[VisibleSign], patient: PatientConfig, view: str = "patient_front"
    ) -> list[str]:
        """Return a list of frame references (data URIs or file paths)."""
        ...

    @property
    def is_photoreal(self) -> bool: ...


class StubSceneRenderer:
    """Renders a labelled placeholder frame encoding present signs."""

    def __init__(self, sign_lib: SignRenderLibrary | None = None) -> None:
        self._sign_lib = sign_lib or SignRenderLibrary.load()
        self._warned = False

    @property
    def is_photoreal(self) -> bool:
        return False

    def render(
        self, signs: list[VisibleSign], patient: PatientConfig, view: str = "patient_front"
    ) -> list[str]:
        if not self._warned:
            logger.warning(
                "StubSceneRenderer active — placeholder frames only, NOT photoreal. "
                "Do not use frame_stream C3 results from stub renders for publication."
            )
            self._warned = True
        try:
            return [self._render_pillow(signs, patient, view)]
        except Exception as e:
            logger.debug(f"Pillow unavailable ({e}); emitting text-encoded frame")
            return [self._render_text_frame(signs)]

    def _render_pillow(self, signs, patient, view) -> str:
        from io import BytesIO

        from PIL import Image, ImageDraw  # type: ignore[import-untyped]

        img = Image.new("RGB", (512, 512), (235, 225, 215))
        draw = ImageDraw.Draw(img)
        # Base "face" oval
        draw.ellipse([156, 96, 356, 356], fill=(232, 205, 190), outline=(120, 100, 90))
        # Encode signs as coloured overlays (deterministic positions)
        for i, sign in enumerate(signs):
            rp = self._sign_lib.resolve(sign.sign_id, sign.severity)
            color = (200, 60, 60)
            if rp and rp.skin_color_rgb:
                color = tuple(rp.skin_color_rgb)  # type: ignore[assignment]
            y = 120 + (i % 6) * 40
            draw.rectangle([380, y, 500, y + 30], fill=color)
            draw.text((384, y + 8), sign.sign_id[:16], fill=(20, 20, 20))
        draw.text((20, 20), f"view={view} signs={len(signs)}", fill=(20, 20, 20))

        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def _render_text_frame(self, signs) -> str:
        """Absolute fallback: a text descriptor (clearly marked non-visual)."""
        ids = ",".join(s.sign_id for s in signs)
        return f"text-frame://signs={ids or 'none'}"


class GodotSceneRenderer:
    """Drives a Godot 4 headless process to render avatar frames."""

    # Files that must exist for the Godot project to be runnable.
    REQUIRED_FILES = [
        "project.godot",
        "scenes/render_scene.tscn",
        "scripts/render_main.gd",
        "scripts/procedural_avatar.gd",
        "data/sign_render_map.json",
    ]

    def __init__(self, godot_bin: Path, project_dir: Path, out_dir: Path) -> None:
        self._godot = godot_bin
        self._project = project_dir
        self._out = out_dir
        self._out.mkdir(parents=True, exist_ok=True)

    @property
    def is_photoreal(self) -> bool:
        return True

    def build_command(self, sign_ids: str, view: str, out_file: Path) -> list[str]:
        """The exact CLI the GDScript render_main.gd parses (see project README)."""
        return [
            str(self._godot), "--headless", "--path", str(self._project),
            "--", "--signs", sign_ids, "--view", view, "--out", str(out_file),
        ]

    def validate(self) -> tuple[bool, list[str]]:
        """Check the Godot project has all required files. (Does not run Godot.)"""
        missing = [f for f in self.REQUIRED_FILES if not (self._project / f).exists()]
        return (not missing), missing

    def render(
        self, signs: list[VisibleSign], patient: PatientConfig, view: str = "patient_front"
    ) -> list[str]:
        sign_ids = ",".join(s.sign_id for s in signs)
        digest = hashlib.sha1(f"{sign_ids}|{view}".encode()).hexdigest()[:12]
        out_file = self._out / f"frame_{digest}.png"
        if out_file.exists():  # deterministic cache
            return [str(out_file)]
        cmd = self.build_command(sign_ids, view, out_file)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            if out_file.exists():
                return [str(out_file)]
            logger.error("Godot exited 0 but produced no frame; falling back to stub")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Godot render failed: {e}; falling back to stub")
        return StubSceneRenderer().render(signs, patient, view)


def godot_project_dir() -> Path:
    return settings.resources_dir / "avatars_cc0" / "godot_project"


def get_scene_renderer() -> SceneRenderer:
    """Factory: Godot renderer if a binary + a valid project are present, else stub."""
    if settings.godot_bin and Path(settings.godot_bin).exists():
        project = godot_project_dir()
        out = settings.resources_dir / "_render_cache"
        if (project / "project.godot").exists():
            renderer = GodotSceneRenderer(Path(settings.godot_bin), project, out)
            ok, missing = renderer.validate()
            if ok:
                return renderer
            logger.warning(f"Godot project incomplete (missing {missing}); using stub")
        else:
            logger.warning(f"Godot bin set but no project.godot at {project}; using stub")
    return StubSceneRenderer()
