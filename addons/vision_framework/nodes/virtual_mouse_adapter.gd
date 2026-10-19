extends Node
class_name VirtualMouseAdapter

@export var enabled: bool = true
@export var screen_scale: Vector2 = Vector2(1920, 1080)
@export var smoothing: float = 0.2

var last_mouse_pos: Vector2 = Vector2.ZERO

func _ready() -> void:
	if Engine.is_editor_hint():
		return
	# Ensure VisionBridge singleton is ready before connecting
	if VisionBridge and VisionBridge.has_signal("landmark_updated"):
		VisionBridge.connect("landmark_updated", _on_landmark_updated)

func _on_landmark_updated(hand_id: int, landmarks: Array) -> void:
	if not enabled or landmarks.is_empty():
		return
		
	# Use Index Finger Tip (Landmark 8) to drive mouse coordinates
	var index_tip = landmarks[8]
	# MediaPipe gives normalized coordinates (0.0 to 1.0). Invert X since frame is mirrored.
	var target_x = (1.0 - index_tip["x"]) * screen_scale.x
	var target_y = index_tip["y"] * screen_scale.y
	
	var target_pos = Vector2(target_x, target_y)
	last_mouse_pos = last_mouse_pos.lerp(target_pos, smoothing)
	
	# Move the mouse position using Input.warp_mouse
	Input.warp_mouse(last_mouse_pos)
