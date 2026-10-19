import mediapipe as mp
import math

class HandTracker:
    def __init__(self, max_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process(self, frame):
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label # Left or Right
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
                
                # Simple Pinch Detection (Distance between Thumb Tip [4] and Index Tip [8])
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                is_pinching = pinch_dist < 0.05
                
                hands_data.append({
                    "id": idx,
                    "handedness": handedness,
                    "landmarks": landmarks,
                    "is_pinching": is_pinching,
                    "pinch_distance": pinch_dist
                })
                
        return hands_data