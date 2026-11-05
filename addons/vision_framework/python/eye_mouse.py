import cv2
import mediapipe as mp
import pyautogui
import math
from collections import deque
import time

# Disable PyAutoGUI failsafe to prevent crashes
pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# Smoothing buffers for ratios
ratio_history_length = 10
h_ratio_buffer = deque(maxlen=ratio_history_length)
v_ratio_buffer = deque(maxlen=ratio_history_length)

# Mouse position smoothing buffer
x_buffer = deque(maxlen=7)
y_buffer = deque(maxlen=7)

# Auto-Calibration Variables
calibrated = False
calib_start_time = time.time()
calib_duration = 5.0  # 5 seconds calibration time

min_h_ratio, max_h_ratio = 999.0, -999.0
min_v_ratio, max_v_ratio = 999.0, -999.0
h_min, h_max, v_min, v_max = 0.35, 0.65, 0.35, 0.65

print("[EyeTracker] Calibration starting... Please look around the corners and center of your screen for 5 seconds.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Eye boundaries and center calculation
        outer = landmarks[33]
        inner = landmarks[133]
        top = landmarks[159]
        bottom = landmarks[145]

        eye_center_x = (outer.x + inner.x) / 2.0
        eye_center_y = (top.y + bottom.y) / 2.0

        eye_width = abs(inner.x - outer.x)
        eye_height = abs(bottom.y - top.y)

        iris_a = landmarks[468]
        iris_b = landmarks[473]
        
        dist_a = math.hypot(iris_a.x - eye_center_x, iris_a.y - eye_center_y)
        dist_b = math.hypot(iris_b.x - eye_center_x, iris_b.y - eye_center_y)
        iris = iris_a if dist_a < dist_b else iris_b

        if eye_width > 0 and eye_height > 0:
            raw_h_ratio = (iris.x - min(outer.x, inner.x)) / eye_width
            raw_v_ratio = (iris.y - min(top.y, bottom.y)) / eye_height

            # Smooth the ratios first
            h_ratio_buffer.append(raw_h_ratio)
            v_ratio_buffer.append(raw_v_ratio)
            h_ratio = sum(h_ratio_buffer) / len(h_ratio_buffer)
            v_ratio = sum(v_ratio_buffer) / len(v_ratio_buffer)

            if not calibrated:
                # Calibration phase: gather min/max bounds of smoothed eye ratios
                min_h_ratio = min(min_h_ratio, h_ratio)
                max_h_ratio = max(max_h_ratio, h_ratio)
                min_v_ratio = min(min_v_ratio, v_ratio)
                max_v_ratio = max(max_v_ratio, v_ratio)

                elapsed = time.time() - calib_start_time
                remaining = int(calib_duration - elapsed)
                
                # Display calibration message on the webcam preview window
                cv2.putText(frame, f"CALIBRATING: Look around ({max(0, remaining)}s)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if elapsed >= calib_duration:
                    calibrated = True
                    h_margin = (max_h_ratio - min_h_ratio) * 0.05
                    v_margin = (max_v_ratio - min_v_ratio) * 0.05
                    
                    h_min = min_h_ratio + h_margin
                    h_max = max_h_ratio - h_margin
                    v_min = min_v_ratio + v_margin
                    v_max = max_v_ratio - v_margin
                    print(f"[EyeTracker] Calibration Successful! H: [{h_min:.3f}, {h_max:.3f}] V: [{v_min:.3f}, {v_max:.3f}]")
            else:
                # Normal Operation Mode (using calibrated boundaries)
                norm_x = (h_ratio - h_min) / (h_max - h_min) if (h_max - h_min) > 0 else 0.5
                norm_y = (v_ratio - v_min) / (v_max - v_min) if (v_max - v_min) > 0 else 0.5

                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = 1.0 - max(0.0, min(1.0, norm_y))  # Inverted vertical mapping

                target_x = norm_x * SCREEN_W
                target_y = norm_y * SCREEN_H

                # --- RADIAL DISTANCE LOCK (Stops orbiting loops) ---
                if len(x_buffer) > 0:
                    last_x = sum(x_buffer) / len(x_buffer)
                    last_y = sum(y_buffer) / len(y_buffer)
                    
                    # Calculate straight-line (Euclidean) distance from current position
                    radial_distance = math.hypot(target_x - last_x, target_y - last_y)
                    
                    # If movement is within a 15-pixel radius, freeze position completely
                    if radial_distance < 15.0:
                        target_x = last_x
                        target_y = last_y

                # Append to mouse coordinate buffer and smooth
                x_buffer.append(target_x)
                y_buffer.append(target_y)

                smoothed_x = sum(x_buffer) / len(x_buffer)
                smoothed_y = sum(y_buffer) / len(y_buffer)

                pyautogui.moveTo(int(smoothed_x), int(smoothed_y), _pause=False)

        # Green tracking dots on both eyes
        left_iris_lm = landmarks[468]   
        right_iris_lm = landmarks[473]  
        cv2.circle(frame, (int(left_iris_lm.x * w_frame), int(left_iris_lm.y * h_frame)), 6, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_iris_lm.x * w_frame), int(right_iris_lm.y * h_frame)), 6, (0, 255, 0), -1)

        # Click on left eye blink
        other_top = landmarks[386]
        other_bottom = landmarks[374]
        eye_opening = math.hypot(other_top.x - other_bottom.x, other_top.y - other_bottom.y)
        
        if eye_opening < 0.012:
            pyautogui.click()
            pyautogui.sleep(0.4) 

    cv2.imshow("Eye Tracker Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()