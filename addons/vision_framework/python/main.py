import asyncio
import cv2
import json
import math
import os
import sys
import time
import websockets
import mediapipe as mp
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

# ==================== MEDIAPIPE SETUP ====================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

THUMB_TIP = 4
INDEX_FINGER_TIP = 8

class WASDController:
    def __init__(self):
        self.calib_x = 0.5
        self.calib_y = 0.5
        self.is_calibrated = False
        self.prev_pinching = False
        self.direction = "STOP"
        self.latest_payload = {"type": "frame_data", "hands": []}
        self.is_running = True

controller = WASDController()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

# ==================== CAMERA LOOP ====================
async def camera_loop():
    while controller.is_running:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        controller.direction = "STOP"
        is_pinching = False
        hand_detected = False
        raw_landmarks_list = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                hand_detected = True
                raw_landmarks_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]

                thumb = hand_landmarks[THUMB_TIP]
                index = hand_landmarks[INDEX_FINGER_TIP]

                pinch_dist_px = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
                is_pinching = (pinch_dist_px / min(w, h)) < 0.09

                # Lock calibration origin ON INITIAL PINCH TRANSITION ONLY
                if is_pinching and not controller.prev_pinching:
                    controller.calib_x = index.x
                    controller.calib_y = index.y
                    controller.is_calibrated = True

                controller.prev_pinching = is_pinching

                # Process WASD Direction WHILE PINCH IS HELD
                if is_pinching and controller.is_calibrated:
                    px_diff_x = (index.x - controller.calib_x) * w
                    px_diff_y = (index.y - controller.calib_y) * h

                    distance = math.hypot(px_diff_x, px_diff_y)
                    dead_zone_radius = int(DEAD_ZONE * min(w, h))

                    if distance < dead_zone_radius:
                        controller.direction = "STOP"
                    elif abs(px_diff_x) > abs(px_diff_y):
                        controller.direction = "LEFT" if px_diff_x < 0 else "RIGHT"
                    else:
                        controller.direction = "UP" if px_diff_y < 0 else "DOWN"

                tip_px = (int(index.x * w), int(index.y * h))
                dot_color = (0, 0, 255) if is_pinching else (255, 0, 0)
                cv2.circle(frame, tip_px, 8, dot_color, -1)
                break

        # Map current camera direction to a simulated keyboard vector
        virtual_x, virtual_y = 0.0, 0.0
        if controller.direction == "LEFT": virtual_x = -1.0
        elif controller.direction == "RIGHT": virtual_x = 1.0
        elif controller.direction == "UP": virtual_y = -1.0
        elif controller.direction == "DOWN": virtual_y = 1.0

        if raw_landmarks_list:
            # Inject simulated WASD state directly into Godot via Landmark 0 (Wrist)
            raw_landmarks_list[0]["x"] = virtual_x
            raw_landmarks_list[0]["y"] = virtual_y

        controller.latest_payload = {
            "type": "frame_data",
            "hands": [
                {
                    "hand_id": 0,
                    "id": 0,
                    "handedness": "Right",
                    "is_pinching": is_pinching,
                    "landmarks": raw_landmarks_list
                }
            ] if hand_detected else []
        }

        # Draw WASD Overlay HUD
        if is_pinching and controller.is_calibrated:
            center_px_x = int(controller.calib_x * w)
            center_px_y = int(controller.calib_y * h)
            radius = int(DEAD_ZONE * min(w, h))

            cv2.circle(frame, (center_px_x, center_px_y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (center_px_x, center_px_y), 4, (0, 255, 0), -1)

            offset = int(radius * 0.707)
            cv2.line(frame, (center_px_x - offset, center_px_y - offset),
                     (center_px_x + offset, center_px_y + offset), (0, 140, 0), 1)
            cv2.line(frame, (center_px_x + offset, center_px_y - offset),
                     (center_px_x - offset, center_px_y + offset), (0, 140, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            w_col = (0, 255, 255) if controller.direction == "UP" else (0, 255, 0)
            s_col = (0, 255, 255) if controller.direction == "DOWN" else (0, 255, 0)
            a_col = (0, 255, 255) if controller.direction == "LEFT" else (0, 255, 0)
            d_col = (0, 255, 255) if controller.direction == "RIGHT" else (0, 255, 0)

            cv2.putText(frame, "W", (center_px_x - 8, center_px_y - radius - 6), font, 0.6, w_col, 2)
            cv2.putText(frame, "S", (center_px_x - 6, center_px_y + radius + 18), font, 0.6, s_col, 2)
            cv2.putText(frame, "A", (center_px_x - radius - 18, center_px_y + 6), font, 0.6, a_col, 2)
            cv2.putText(frame, "D", (center_px_x + radius + 6, center_px_y + 6), font, 0.6, d_col, 2)

            status_col = (0, 255, 255) if controller.direction != "STOP" else (0, 255, 0)
            cv2.putText(frame, f"WASD: {controller.direction}", (15, 25), font, 0.6, status_col, 2)
        else:
            cv2.putText(frame, "Pinch and hold to activate WASD Joystick", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("WASD Camera Feed", frame)
        cv2.waitKey(1)  # Required to keep the OpenCV window responsive

        await asyncio.sleep(0.005)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

# ==================== WEBSOCKET SERVER ====================
async def vision_handler(websocket):
    print(f"[VisionService] Godot client connected.")
    try:
        while controller.is_running:
            await websocket.send(json.dumps(controller.latest_payload))
            await asyncio.sleep(0.016)
    except websockets.exceptions.ConnectionClosed:
        print("[VisionService] Godot client disconnected. Shutting down camera...")
        controller.is_running = False
    except Exception as e:
        print(f"[VisionService Error] {e}")
        controller.is_running = False

async def main():
    asyncio.create_task(camera_loop())
    async with websockets.serve(vision_handler, HOST, PORT):
        print(f"[VisionService] WebSocket server listening on ws://{HOST}:{PORT}")
        while controller.is_running:
            await asyncio.sleep(0.1)
    sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("[VisionService] Shutdown complete.")