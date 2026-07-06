extends Node3D
## Headless CliniCraft avatar renderer — entry point.
##
## Invoked by clinicraft.render.scene_renderer.GodotSceneRenderer as:
##   godot4 --headless --path <project> -- \
##          --signs pallor,diaphoresis --view patient_front --out /abs/frame.png
##
## Parses the user args, builds a scene inside an offscreen SubViewport, applies
## the requested clinical signs to a procedural avatar, renders one 512x512
## frame and saves it as PNG, then quits.
##
## STATUS: structurally-real skeleton. Verified for syntax/contract against the
## Godot 4.2 API but NOT executed in CI (no Godot binary / GPU available in the
## build environment). Headless pixel output requires a Godot 4 runtime with a
## rendering backend (GPU, or software rasteriser via e.g. xvfb-run). See
## ../README.md for how to verify.

const ProceduralAvatar := preload("res://scripts/procedural_avatar.gd")

const FRAME_SIZE := Vector2i(512, 512)

var _signs: PackedStringArray = PackedStringArray()
var _view: String = "patient_front"
var _out: String = "frame.png"
var _subviewport: SubViewport


func _ready() -> void:
	_parse_args()
	_build_scene()
	# Let the SubViewport draw before capturing.
	_subviewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	_capture_and_save()
	get_tree().quit()


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		match args[i]:
			"--signs":
				if i + 1 < args.size():
					_signs = args[i + 1].split(",", false)
					i += 1
			"--view":
				if i + 1 < args.size():
					_view = args[i + 1]
					i += 1
			"--out":
				if i + 1 < args.size():
					_out = args[i + 1]
					i += 1
		i += 1
	print("[render_main] signs=%s view=%s out=%s" % [_signs, _view, _out])


func _build_scene() -> void:
	_subviewport = SubViewport.new()
	_subviewport.size = FRAME_SIZE
	_subviewport.transparent_bg = false
	_subviewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	add_child(_subviewport)

	# Environment (neutral clinical backdrop + ambient light)
	var world_env := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.93, 0.91, 0.89)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.55, 0.55, 0.55)
	world_env.environment = env
	_subviewport.add_child(world_env)

	# Key light
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-45.0, -30.0, 0.0)
	light.light_energy = 1.2
	_subviewport.add_child(light)

	# Camera
	var cam := Camera3D.new()
	cam.position = _camera_position_for_view(_view)
	cam.look_at(Vector3(0.0, 1.1, 0.0), Vector3.UP)
	_subviewport.add_child(cam)
	cam.make_current()

	# Avatar + signs
	var avatar := ProceduralAvatar.new()
	_subviewport.add_child(avatar)
	avatar.load_sign_map("res://data/sign_render_map.json")
	for s in _signs:
		avatar.apply_sign(s)


func _camera_position_for_view(view: String) -> Vector3:
	match view:
		"close_up_face":
			return Vector3(0.0, 1.55, 0.9)
		"full_body":
			return Vector3(0.0, 1.0, 3.4)
		_:  # patient_front (default)
			return Vector3(0.0, 1.25, 2.3)


func _capture_and_save() -> void:
	var img := _subviewport.get_texture().get_image()
	if img == null:
		push_error("[render_main] null viewport image — no rendering backend?")
		return
	var dir := _out.get_base_dir()
	if dir != "" and not DirAccess.dir_exists_absolute(dir):
		DirAccess.make_dir_recursive_absolute(dir)
	var err := img.save_png(_out)
	if err != OK:
		push_error("[render_main] save_png failed for %s (err %d)" % [_out, err])
	else:
		print("[render_main] saved %s" % _out)
