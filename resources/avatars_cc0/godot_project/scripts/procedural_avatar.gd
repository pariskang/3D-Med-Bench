extends Node3D
## Low-fidelity procedural humanoid for the CliniCraft Tier-A render skeleton.
##
## Clinical signs are applied as material / posture / effect changes so that each
## sign yields a visually distinguishable frame. This is deliberately NOT
## photoreal — it is a runnable placeholder that exercises the render→capture→
## multimodal-judge path. The Tier-B upgrade is a rigged MetaHuman / GLB with
## morph targets: when such an asset is present, blendshape signs drive real
## morph targets instead of the approximations below (see README).
##
## Sign parameters are read from res://data/sign_render_map.json, which is
## generated from resources/sign_render_lib/signs.yaml by
## clinicraft.render.sign_map_export — a single source of truth shared with the
## Python-side Sign Rendering Library.

var _sign_map: Dictionary = {}

var _skin := Color(0.86, 0.68, 0.60)      # default skin albedo
var _body_parts: Array[MeshInstance3D] = []
var _head: MeshInstance3D
var _sweat: GPUParticles3D


func _init() -> void:
	_build_body()


func _build_body() -> void:
	# Torso, head, arms, legs from primitives.
	_head = _add_sphere(Vector3(0.0, 1.55, 0.0), 0.13)
	_add_capsule(Vector3(0.0, 0.95, 0.0), 0.55, 0.20)          # torso
	_add_capsule(Vector3(-0.28, 1.02, 0.0), 0.48, 0.065)       # left arm
	_add_capsule(Vector3(0.28, 1.02, 0.0), 0.48, 0.065)        # right arm
	_add_capsule(Vector3(-0.12, 0.35, 0.0), 0.62, 0.085)       # left leg
	_add_capsule(Vector3(0.12, 0.35, 0.0), 0.62, 0.085)        # right leg
	_refresh_skin()


func load_sign_map(path: String) -> void:
	if not FileAccess.file_exists(path):
		push_warning("[avatar] sign map not found: %s" % path)
		return
	var f := FileAccess.open(path, FileAccess.READ)
	var parsed = JSON.parse_string(f.get_as_text())
	if typeof(parsed) == TYPE_DICTIONARY:
		_sign_map = parsed


func apply_sign(sign_id: String) -> void:
	var entry: Dictionary = _sign_map.get(sign_id, {})
	if entry.is_empty():
		push_warning("[avatar] unknown sign '%s'" % sign_id)
		return

	# 1. Skin colour signs (pallor / cyanosis / jaundice / sallow ...).
	if entry.has("skin_color_rgb"):
		var c: Array = entry["skin_color_rgb"]
		_skin = Color(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
		_refresh_skin()

	# 2. Posture signs (orthopnea / tripod / decorticate ...).
	if entry.has("posture"):
		_apply_posture(String(entry["posture"]))

	# 3. Diaphoresis → sweat particle sheen on the face.
	if sign_id == "diaphoresis":
		_add_sweat()

	# 4. Blendshape / animation signs need a rigged mesh; record for parity.
	if entry.has("blendshapes"):
		print("[avatar] '%s' → blendshapes %s (needs rigged Tier-B asset)"
			% [sign_id, entry["blendshapes"]])
	if entry.has("animation"):
		print("[avatar] '%s' → animation '%s' (needs rigged Tier-B asset)"
			% [sign_id, entry["animation"]])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

func _add_capsule(pos: Vector3, height: float, radius: float) -> MeshInstance3D:
	var mesh := CapsuleMesh.new()
	mesh.height = height
	mesh.radius = radius
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.position = pos
	add_child(mi)
	_body_parts.append(mi)
	return mi


func _add_sphere(pos: Vector3, radius: float) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.position = pos
	add_child(mi)
	_body_parts.append(mi)
	return mi


func _refresh_skin() -> void:
	for part in _body_parts:
		var mat := StandardMaterial3D.new()
		mat.albedo_color = _skin
		mat.roughness = 0.7
		part.material_override = mat


func _apply_posture(posture: String) -> void:
	match posture:
		"orthopnea", "tripod":
			# lean forward
			rotation_degrees = Vector3(18.0, 0.0, 0.0)
		"decorticate":
			rotation_degrees = Vector3(0.0, 0.0, 0.0)
		"decerebrate":
			rotation_degrees = Vector3(-6.0, 0.0, 0.0)
		"wide_stance", "requires_support":
			scale = Vector3(1.1, 0.98, 1.0)
		_:
			pass


func _add_sweat() -> void:
	_sweat = GPUParticles3D.new()
	_sweat.amount = 24
	_sweat.lifetime = 1.5
	_sweat.position = _head.position
	var pm := ParticleProcessMaterial.new()
	pm.gravity = Vector3(0.0, -0.6, 0.0)
	pm.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE
	pm.emission_sphere_radius = 0.14
	_sweat.process_material = pm
	# Small droplet mesh so the particles are actually drawn.
	var droplet := SphereMesh.new()
	droplet.radius = 0.008
	droplet.height = 0.016
	var dmat := StandardMaterial3D.new()
	dmat.albedo_color = Color(0.8, 0.85, 0.95, 0.7)
	droplet.material = dmat
	_sweat.draw_pass_1 = droplet
	add_child(_sweat)
	print("[avatar] diaphoresis sweat sheen applied")
