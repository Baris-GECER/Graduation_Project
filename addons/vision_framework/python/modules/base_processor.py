import cv2
import json
import os

class BaseProcessor:
    def __init__(self, config_path=None):
        if config_path is None:
            # Automatically resolve config.json relative to this file's location 
            # (addons/vision_framework/python/modules/ -> goes up one level to python/config.json)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "..", "config.json")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        cam_cfg = self.config["camera"]
        device_id = cam_cfg["device_id"]
        
        # Try opening camera with explicit backend fallback for Windows/Mac/Linux
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(device_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

        if not self.cap.isOpened():
            print(f"[BaseProcessorError] Could not open webcam index {device_id}!")

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        success, frame = self.cap.read()
        if not success:
            return None
        # Flip frame horizontally for a natural mirror feel
        frame = cv2.flip(frame, 1)
        return frame

    def show_debug_window(self, frame, hands_data):
        for hand in hands_data:
            pinching = hand["is_pinching"]
            color = (0, 0, 255) if pinching else (0, 255, 0)
            text = f"Hand: {hand['handedness']} | Pinch: {pinching}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Python Vision Debug Feed", frame)
        cv2.waitKey(1)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()