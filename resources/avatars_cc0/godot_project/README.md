# CliniCraft Avatar Renderer — Godot 4 (Tier A)

A **headless render tool** that draws a 3D patient avatar with clinical signs
applied, for CliniCraft-Bench's C3 (3D active-perception) dimension. The Python
side (`clinicraft/render/scene_renderer.py::GodotSceneRenderer`) shells out to
Godot with this project to produce PNG frames for the multimodal judge.

## ⚠️ Status: structurally-real skeleton — requires a Godot 4 runtime to verify

This project is **written against the Godot 4.2 API and wired to the exact CLI
contract the Python renderer uses**, but it has **NOT been executed in this
repo's CI** — the build environment has no Godot binary and no GPU. Producing
actual pixels requires a Godot 4 runtime with a rendering backend.

What *is* verified here (by `tests/test_render.py`):
- the project structure and required files exist;
- the sign→render map (`data/sign_render_map.json`) is generated from, and stays
  consistent with, `resources/sign_render_lib/signs.yaml`;
- `GodotSceneRenderer` builds the correct command line and falls back to the
  stub renderer when Godot or this project is absent.

What still needs a Godot runtime to confirm: that headless rendering actually
emits a non-empty PNG on the target machine.

## Fidelity

Tier A here is a **low-fidelity procedural avatar** (primitive head/torso/limbs).
Signs are applied as:
- **skin colour** — pallor, cyanosis, jaundice, sallow complexion (real, works today)
- **posture** — orthopnea / tripod lean, decerebrate/decorticate (real)
- **diaphoresis** — sweat particle sheen (real)
- **blendshape / animation** signs (ptosis, facial palsy, ataxic gait, tremor) —
  logged but **require a rigged mesh with morph targets** (Tier B). Drop a rigged
  `avatar.glb` with the named morph targets in `assets/` and extend
  `procedural_avatar.gd::apply_sign` to drive them.

Tier B (photoreal UE5/MetaHuman) is out of scope for automatic per-case scoring
per the protocol (§6) — it is for demo / human-comparison only.

## Run it (on a machine with Godot 4.2+)

```bash
# software rendering via xvfb (Linux, no GPU):
xvfb-run -a godot4 --headless --path . -- \
    --signs pallor,diaphoresis --view patient_front --out /tmp/frame.png

# or with a GPU:
godot4 --headless --path . -- --signs cyanosis --view close_up_face --out /tmp/c.png
```

Then point CliniCraft at it:

```bash
export CLINICRAFT_GODOT_BIN=$(command -v godot4)
# GodotSceneRenderer activates automatically when the binary + this project exist
python -m pytest tests/test_render.py -k godot -q
```

## Regenerate the sign map after editing signs.yaml

```bash
python -c "from clinicraft.render.sign_map_export import export_sign_map; print(export_sign_map())"
```

## Files

```
godot_project/
├── project.godot                 # Godot 4.2 config (GL Compatibility, 512x512)
├── scenes/render_scene.tscn      # main scene → render_main.gd
├── scripts/render_main.gd        # headless entry: parse args → build → capture → save PNG
├── scripts/procedural_avatar.gd  # primitive humanoid + apply_sign()
└── data/sign_render_map.json     # generated from signs.yaml (single source of truth)
```
