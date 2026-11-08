import asyncio
import cv2
import json
import math
import os
import sys
import time
import websockets
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================== CONFIGURATION & SETTINGS ====================
HOST = "127.0.0.1"
PORT = 8765

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: File not found at {MODEL_PATH}")

WINDOW_W, WINDOW_H = 640, 360
DEAD_ZONE = 0.08

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# ==================== MEDIAPIPE SETUP ====================
# 1. Hands Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(hand_options)
THUMB_TIP, INDEX_FINGER_TIP = 4, 8

# 2. Face Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==================== STATE CONTROLLERS ====================
class WASDController:
    def __init__(self):
        self.calib_x, self.calib_y = 0.5, 0.5
        self.is_joystick_active = False
        self.prev_pinching = False
        self.direction = "STOP"
        self.latest_payload = {"type": "frame_data", "hands": []}
        self.is_running = True

class HeadMouseController:
    def __init__(self):
        self.calibrated = False
        self.calib_start_time = time.time()
        self.calib_duration = 5.0
        self.min_x, self.max_x = float("inf"), float("-inf")
        self.min_y, self.max_y = float("inf"), float("-inf")
        self.x_min_bound, self.x_max_bound = 0.4, 0.6
        self.y_min_bound, self.y_max_bound = 0.4, 0.6
        self.prev_target_x, self.prev_target_y = pyautogui.position()
        
        # === TUNING PARAMETERS ===
        self.smoothing = 0.88          # Higher = smoother (keeps anti-shake)
        self.sensitivity = 1.00        # 1.0 = full screen range (was the problem)
        self.center_deadzone = 0.032   # Still ignores tiny movements near center
        
        self.blink_threshold = 0.018
        self.wink_latched = False
        self.last_click_time = 0.0
        self.click_cooldown = 0.5

wasd_ctrl = WASDController()
face_ctrl = HeadMouseController()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

print("[VisionService] Starting unified Hand and Face tracking...")

# ==================== CAMERA LOOP ====================
async def camera_loop():
    while wasd_ctrl.is_running:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000)
        
        # --- PROCESS HANDS (WASD) ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        wasd_ctrl.direction = "STOP"
        is_pinching = False
        hand_detected = False
        raw_landmarks_list = []

        if hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                hand_detected = True
                raw_landmarks_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]

                thumb = hand_landmarks[THUMB_TIP]
                index = hand_landmarks[INDEX_FINGER_TIP]
                pinch_dist_px = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
                is_pinching = (pinch_dist_px / min(w, h)) < 0.09

                if is_pinching and not wasd_ctrl.prev_pinching:
                    wasd_ctrl.is_joystick_active = not wasd_ctrl.is_joystick_active
                    if wasd_ctrl.is_joystick_active:
                        wasd_ctrl.calib_x, wasd_ctrl.calib_y = index.x, index.y
                wasd_ctrl.prev_pinching = is_pinching

                if wasd_ctrl.is_joystick_active:
                    px_diff_x = (index.x - wasd_ctrl.calib_x) * w
                    px_diff_y = (index.y - wasd_ctrl.calib_y) * h
                    distance = math.hypot(px_diff_x, px_diff_y)
                    dead_zone_radius = int(DEAD_ZONE * min(w, h))

                    if distance < dead_zone_radius:
                        wasd_ctrl.direction = "STOP"
                    elif abs(px_diff_x) > abs(px_diff_y):
                        wasd_ctrl.direction = "LEFT" if px_diff_x < 0 else "RIGHT"
                    else:
                        wasd_ctrl.direction = "UP" if px_diff_y < 0 else "DOWN"

                tip_px = (int(index.x * w), int(index.y * h))
                cv2.circle(frame, tip_px, 8, (0, 0, 255) if is_pinching else (255, 0, 0), -1)
                break

        # Map current camera direction to a simulated keyboard vector
        virtual_x, virtual_y = 0.0, 0.0
        if wasd_ctrl.direction == "LEFT": virtual_x = -1.0
        elif wasd_ctrl.direction == "RIGHT": virtual_x = 1.0
        elif wasd_ctrl.direction == "UP": virtual_y = -1.0
        elif wasd_ctrl.direction == "DOWN": virtual_y = 1.0

        if raw_landmarks_list:
            raw_landmarks_list[0]["x"] = virtual_x
            raw_landmarks_list[0]["y"] = virtual_y

        wasd_ctrl.latest_payload = {
            "type": "frame_data",
            "hands": [{
                "hand_id": 0, "id": 0, "handedness": "Right", "is_pinching": is_pinching,
                "landmarks": raw_landmarks_list
            }] if hand_detected else []
        }

        # --- PROCESS FACE (HEAD MOUSE & CLICKS) ---
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            f_marks = face_result.multi_face_landmarks[0].landmark
            
            # Head Tracking
            nose = f_marks[4]
            nx, ny = nose.x, nose.y

            if not face_ctrl.calibrated:
                face_ctrl.min_x = min(face_ctrl.min_x, nx)
                face_ctrl.max_x = max(face_ctrl.max_x, nx)
                face_ctrl.min_y = min(face_ctrl.min_y, ny)
                face_ctrl.max_y = max(face_ctrl.max_y, ny)

                elapsed = time.time() - face_ctrl.calib_start_time
                remaining = max(0, int(face_ctrl.calib_duration - elapsed))
                cv2.putText(frame, f"CALIBRATING HEAD: Look corners ({remaining}s)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if elapsed >= face_ctrl.calib_duration:
                    margin_x = (face_ctrl.max_x - face_ctrl.min_x) * 0.08   # slightly less aggressive margin
                    margin_y = (face_ctrl.max_y - face_ctrl.min_y) * 0.08
                    face_ctrl.x_min_bound = face_ctrl.min_x + margin_x
                    face_ctrl.x_max_bound = face_ctrl.max_x - margin_x
                    face_ctrl.y_min_bound = face_ctrl.min_y + margin_y
                    face_ctrl.y_max_bound = face_ctrl.max_y - margin_y
                    face_ctrl.calibrated = True
                    print("[VisionService] Head Mouse Calibration Complete.")
            else:
                # Normalize
                norm_x = (nx - face_ctrl.x_min_bound) / (face_ctrl.x_max_bound - face_ctrl.x_min_bound)
                norm_y = (ny - face_ctrl.y_min_bound) / (face_ctrl.y_max_bound - face_ctrl.y_min_bound)
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                # Dead zone (only near center)
                dx = norm_x - 0.5
                dy = norm_y - 0.5
                if abs(dx) < face_ctrl.center_deadzone:
                    norm_x = 0.5
                if abs(dy) < face_ctrl.center_deadzone:
                    norm_y = 0.5

                # Sensitivity (now 1.0 so full range is available)
                norm_x = 0.5 + (norm_x - 0.5) * face_ctrl.sensitivity
                norm_y = 0.5 + (norm_y - 0.5) * face_ctrl.sensitivity
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                target_x = norm_x * (SCREEN_W - 1)
                target_y = norm_y * (SCREEN_H - 1)

                # Smoothing
                alpha = 1.0 - face_ctrl.smoothing
                target_x = alpha * target_x + face_ctrl.smoothing * face_ctrl.prev_target_x
                target_y = alpha * target_y + face_ctrl.smoothing * face_ctrl.prev_target_y
                
                face_ctrl.prev_target_x, face_ctrl.prev_target_y = target_x, target_y
                pyautogui.moveTo(int(target_x), int(target_y), _pause=False)

            # Wink Clicks
            l_top, l_bottom = f_marks[159], f_marks[145]
            r_top, r_bottom = f_marks[386], f_marks[374]
            left_eye_opening = math.hypot(l_top.x - l_bottom.x, l_top.y - l_bottom.y)
            right_eye_opening = math.hypot(r_top.x - r_bottom.x, r_top.y - r_bottom.y)

            left_is_closed = left_eye_opening < face_ctrl.blink_threshold
            right_is_closed = right_eye_opening < face_ctrl.blink_threshold
            wink_now = (left_is_closed and not right_is_closed) or (right_is_closed and not left_is_closed)

            now = time.time()
            if wink_now:
                if not face_ctrl.wink_latched and now - face_ctrl.last_click_time >= face_ctrl.click_cooldown:
                    pyautogui.click(button="left" if left_is_closed else "right")
                    face_ctrl.last_click_time = now
                face_ctrl.wink_latched = True
            else:
                face_ctrl.wink_latched = False

            # Draw Eye Dots
            if not left_is_closed:
                cv2.circle(frame, (int(f_marks[468].x * w), int(f_marks[468].y * h)), 4, (0, 255, 0), -1)
            if not right_is_closed:
                cv2.circle(frame, (int(f_marks[473].x * w), int(f_marks[473].y * h)), 4, (0, 255, 0), -1)

        # --- DRAW UI OVERLAYS ---
        if wasd_ctrl.is_joystick_active:
            cx, cy = int(wasd_ctrl.calib_x * w), int(wasd_ctrl.calib_y * h)
            radius = int(DEAD_ZONE * min(w, h))
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            offset = int(radius * 0.707)
            cv2.line(frame, (cx - offset, cy - offset), (cx + offset, cy + offset), (0, 140, 0), 1)
            cv2.line(frame, (cx + offset, cy - offset), (cx - offset, cy + offset), (0, 140, 0), 1)

            cv2.putText(frame, "W", (cx - 8, cy - radius - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if wasd_ctrl.direction == "UP" else (0, 255, 0), 2)
            cv2.putText(frame, "S", (cx - 6, cy + radius + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if wasd_ctrl.direction == "DOWN" else (0, 255, 0), 2)
            cv2.putText(frame, "A", (cx - radius - 18, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if wasd_ctrl.direction == "LEFT" else (0, 255, 0), 2)
            cv2.putText(frame, "D", (cx + radius + 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if wasd_ctrl.direction == "RIGHT" else (0, 255, 0), 2)
            cv2.putText(frame, f"WASD: {wasd_ctrl.direction}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if wasd_ctrl.direction != "STOP" else (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Pinch once to toggle WASD Joystick", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Vision Framework Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            wasd_ctrl.is_running = False
            break

        await asyncio.sleep(0.005)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

# ==================== WEBSOCKET SERVER ====================
async def vision_handler(websocket):
    print(f"[VisionService] Godot client connected.")
    try:
        while wasd_ctrl.is_running:
            await websocket.send(json.dumps(wasd_ctrl.latest_payload))
            await asyncio.sleep(0.016)
    except websockets.exceptions.ConnectionClosed:
        print("[VisionService] Godot client disconnected. Shutting down camera...")
        wasd_ctrl.is_running = False
    except Exception as e:
        print(f"[VisionService Error] {e}")
        wasd_ctrl.is_running = False

async def main():
    asyncio.create_task(camera_loop())
    async with websockets.serve(vision_handler, HOST, PORT):
        print(f"[VisionService] WebSocket server listening on ws://{HOST}:{PORT}")
        while wasd_ctrl.is_running:
            await asyncio.sleep(0.1)
    sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        wasd_ctrl.is_running = False
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("[VisionService] Shutdown complete.")