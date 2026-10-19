extends Node2D

@onready var hand_pointer: ColorRect = $HandPointer
@onready var pinch_indicator: ColorRect = $PinchIndicator
@onready var status_label: Label = $UI/StatusLabel

func _ready() -> void:
	# Connect directly to the global VisionBridge Autoload signals
	if Engine.has_singleton("VisionBridge") or get_node_or_null("/root/VisionBridge"):
		VisionBridge.connect("landmark_updated", _on_landmark_updated)
		VisionBridge.connect("hand_detected", _on_hand_detected)
		VisionBridge.connect("gesture_triggered", _on_gesture_triggered)
		status_label.text = "VisionBridge Connected. Camera Active."
	else:
		status_label.text = "Error: VisionBridge Autoload not found!"

func _on_landmark_updated(_hand_id: int, landmarks: Array) -> void:
	if landmarks.size() > 8:
		# Landmark 8 is the Index Finger Tip
		var index_tip = landmarks[8]
		
		# Convert normalized MediaPipe coordinates (0.0 to 1.0) to screen pixel coordinates
		var viewport_size = get_viewport_rect().size
		var target_x = index_tip["x"] * viewport_size.x
		var target_y = index_tip["y"] * viewport_size.y
		
		# Smoothly interpolate the pointer box to the finger position
		var target_pos = Vector2(target_x, target_y)
		hand_pointer.position = hand_pointer.position.lerp(target_pos, 0.4)

func _on_hand_detected(_hand_id: int, handedness: String, is_pinching: bool) -> void:
	# Change color of the pinch indicator box live based on pinch state
	if is_pinching:
		pinch_indicator.color = Color.RED
		status_label.text = "Hand: %s | State: PINCHING!" % handedness
	else:
		pinch_indicator.color = Color.DARK_GRAY
		status_label.text = "Hand: %s | State: Tracking..." % handedness

func _on_gesture_triggered(gesture_name: String) -> void:
	print("Triggered Gesture Event: ", gesture_name)
