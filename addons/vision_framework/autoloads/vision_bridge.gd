extends Node

signal hand_detected(hand_id: int, handedness: String, is_pinching: bool)
signal gesture_triggered(gesture_name: String)
signal landmark_updated(hand_id: int, landmarks: Array)

@export var use_standalone_binary: bool = false
@export var server_port: int = 8765

var socket := WebSocketPeer.new()
var python_process_id: int = -1
var is_connected_to_server := bool(false)
var ws_url := "ws://127.0.0.1:8765"

func _ready() -> void:
	# Enable close request interception to guarantee process termination
	get_tree().set_auto_accept_quit(false)
	
	_launch_python_process()
	_connect_to_server()

func _process(_delta: float) -> void:
	socket.poll()
	var state := socket.get_ready_state()
	
	if state == WebSocketPeer.STATE_OPEN:
		if not is_connected_to_server:
			is_connected_to_server = true
			print("[VisionBridge] Connected to Python Vision Service.")
			
		while socket.get_available_packet_count() > 0:
			var packet := socket.get_packet()
			var text_packet := packet.get_string_from_utf8()
			_parse_packet(text_packet)
			
	elif state == WebSocketPeer.STATE_CLOSING:
		pass
	elif state == WebSocketPeer.STATE_CLOSED:
		if is_connected_to_server:
			is_connected_to_server = false
			print("[VisionBridge] Disconnected from Vision Service.")
			# Attempt reconnect after delay if desired

func _connect_to_server() -> void:
	print("[VisionBridge] Connecting to %s..." % ws_url)
	socket.connect_to_url(ws_url)

func _parse_packet(json_str: String) -> void:
	var json := JSON.new()
	var error := json.parse(json_str)
	if error != OK:
		return
		
	var data = json.get_data()
	if typeof(data) == TYPE_DICTIONARY:
		if data.has("hands"):
			for hand in data["hands"]:
				var h_id: int = hand["id"]
				var h_side: String = hand["handedness"]
				var pinching: bool = hand["is_pinching"]
				var lms: Array = hand["landmarks"]
				
				emit_signal("hand_detected", h_id, h_side, pinching)
				emit_signal("landmark_updated", h_id, lms)
				
				if pinching:
					emit_signal("gesture_triggered", "pinch_" + h_side.to_lower())

func _launch_python_process() -> void:
	var script_path = ProjectSettings.globalize_path("res://addons/vision_framework/python/main.py")
	
	var python_executable = "python"
	var args: PackedStringArray = [script_path]
	
	if use_standalone_binary:
		python_executable = ProjectSettings.globalize_path("res://addons/vision_framework/bin/vision_service.exe")
		args = []

	print("[VisionBridge] Launching Python process...")
	
	# The 3rd parameter (open_console = true) tells Godot to automatically 
	# spawn a visible terminal window on Windows!
	python_process_id = OS.create_process(python_executable, args, true)
	
	if python_process_id == -1:
		push_error("[VisionBridge] Failed to launch Python process! Make sure 'python' is added to your system PATH.")

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST or what == NOTIFICATION_EXIT_TREE:
		_cleanup_process()
		get_tree().quit()

func _cleanup_process() -> void:
	if python_process_id != -1:
		print("[VisionBridge] Terminating Python process (PID: %d)..." % python_process_id)
		if OS.has_feature("windows"):
			OS.execute("taskkill", ["/pid", str(python_process_id), "/f", "/t"])
		else:
			OS.execute("kill", ["-9", str(python_process_id)])
		python_process_id = -1
	socket.close()
