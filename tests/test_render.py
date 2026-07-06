"""
Tests for the Tier-A Godot render skeleton.

These verify everything that does NOT require a Godot runtime: the sign-map
export, the committed map's consistency with signs.yaml, the project structure,
the CLI command contract the GDScript parses, project validation, and the
stub-fallback behaviour. Actual pixel output requires Godot and is out of scope
for CI (see resources/avatars_cc0/godot_project/README.md).
"""

import json
from pathlib import Path

import pytest

from clinicraft.render.sign_map_export import build_sign_map, export_sign_map
from clinicraft.render.scene_renderer import (
    GodotSceneRenderer, StubSceneRenderer, get_scene_renderer, godot_project_dir,
)
from clinicraft.schemas.ground_truth import VisibleSign
from clinicraft.schemas.case_pack import PatientConfig

PROJECT = Path("resources/avatars_cc0/godot_project")


# --------------------------------------------------------------------------
# Sign map export (single source of truth: signs.yaml)
# --------------------------------------------------------------------------

def test_sign_map_export_extracts_render_params():
    m = build_sign_map()
    assert m["pallor"]["skin_color_rgb"] == [220, 195, 180]
    assert m["respiratory_distress"]["posture"] == "orthopnea"
    assert m["cyanosis"]["tier"] == "T1"
    # every sign carries a description + tier
    assert all("tier" in v and "description" in v for v in m.values())


def test_committed_sign_map_is_consistent(tmp_path):
    """The committed JSON must equal a fresh export (or it's stale)."""
    committed = json.loads((PROJECT / "data" / "sign_render_map.json").read_text(encoding="utf-8"))
    fresh = build_sign_map()
    assert committed == fresh, "sign_render_map.json is stale — regenerate via export_sign_map()"


def test_export_writes_into_project(tmp_path):
    out = export_sign_map(out_path=tmp_path / "m.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "diaphoresis" in data


# --------------------------------------------------------------------------
# Project structure & GDScript contract parity
# --------------------------------------------------------------------------

def test_godot_project_structure_complete():
    r = GodotSceneRenderer(Path("/usr/bin/false"), PROJECT, Path("/tmp/_rc"))
    ok, missing = r.validate()
    assert ok, f"missing project files: {missing}"


def test_validate_detects_missing(tmp_path):
    r = GodotSceneRenderer(Path("/usr/bin/false"), tmp_path, tmp_path / "out")
    ok, missing = r.validate()
    assert not ok
    assert "project.godot" in missing


def test_render_main_parses_expected_contract():
    """render_main.gd must parse the exact flags the Python side emits."""
    gd = (PROJECT / "scripts" / "render_main.gd").read_text(encoding="utf-8")
    for flag in ("--signs", "--view", "--out"):
        assert f'"{flag}"' in gd, f"render_main.gd does not parse {flag}"
    assert "OS.get_cmdline_user_args()" in gd
    assert "save_png" in gd


def test_scene_references_main_script():
    tscn = (PROJECT / "scenes" / "render_scene.tscn").read_text(encoding="utf-8")
    assert "res://scripts/render_main.gd" in tscn


def test_avatar_reads_generated_map():
    gd = (PROJECT / "scripts" / "procedural_avatar.gd").read_text(encoding="utf-8")
    assert "res://data/sign_render_map.json" in gd
    assert "apply_sign" in gd


# --------------------------------------------------------------------------
# Command construction & fallback
# --------------------------------------------------------------------------

def test_build_command_matches_contract():
    r = GodotSceneRenderer(Path("/opt/godot4"), PROJECT, Path("/tmp/_rc"))
    cmd = r.build_command("pallor,diaphoresis", "patient_front", Path("/tmp/f.png"))
    assert cmd[:5] == ["/opt/godot4", "--headless", "--path", str(PROJECT), "--"]
    assert cmd[cmd.index("--signs") + 1] == "pallor,diaphoresis"
    assert cmd[cmd.index("--view") + 1] == "patient_front"
    assert cmd[cmd.index("--out") + 1] == "/tmp/f.png"


def test_render_falls_back_to_stub_on_failure(tmp_path, monkeypatch):
    """If Godot errors, render() must still return a (stub) frame, not raise."""
    import clinicraft.render.scene_renderer as sr

    def _boom(*a, **k):
        raise FileNotFoundError("no godot here")
    monkeypatch.setattr(sr.subprocess, "run", _boom)

    r = GodotSceneRenderer(Path("/opt/godot4"), PROJECT, tmp_path / "out")
    frames = r.render([VisibleSign(sign_id="pallor", description="苍白", region="face")],
                      PatientConfig(), "patient_front")
    assert len(frames) == 1                      # stub produced a frame
    assert frames[0].startswith("data:image") or frames[0].startswith("text-frame")


def test_render_falls_back_when_no_output(tmp_path, monkeypatch):
    """Godot exits 0 but writes no PNG → fall back to stub."""
    import clinicraft.render.scene_renderer as sr
    monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: None)  # succeeds, no file

    r = GodotSceneRenderer(Path("/opt/godot4"), PROJECT, tmp_path / "out")
    frames = r.render([VisibleSign(sign_id="cyanosis", description="发绀", region="lips")],
                      PatientConfig(), "patient_front")
    assert len(frames) == 1
    assert frames[0].startswith("data:image") or frames[0].startswith("text-frame")


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def test_factory_returns_stub_without_godot(monkeypatch):
    from clinicraft.config import settings
    monkeypatch.setattr(settings, "godot_bin", None)
    assert isinstance(get_scene_renderer(), StubSceneRenderer)


def test_factory_returns_godot_when_present(tmp_path, monkeypatch):
    from clinicraft.config import settings
    fake_bin = tmp_path / "godot4"
    fake_bin.write_text("#!/bin/sh\n")            # a file that exists
    monkeypatch.setattr(settings, "godot_bin", fake_bin)
    r = get_scene_renderer()
    assert isinstance(r, GodotSceneRenderer)
    ok, _ = r.validate()
    assert ok
