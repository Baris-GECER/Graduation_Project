extends Node
class_name GestureReceiver

signal on_gesture_matched(gesture_name: String)

@export var target_gesture: String = "pinch_right"

func _ready() -> void:
	if Engine.is_editor_hint():
		return
	if VisionBridge.has_signal("gesture_triggered"):
		VisionBridge.connect("gesture_triggered", _on_gesture_triggered)

func _on_gesture_triggered(gesture_name: String) -> void:
	if gesture_name == target_gesture:
		emit_signal("on_gesture_matched", gesture_name)
