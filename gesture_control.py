import cv2
import mediapipe as mp
import numpy as np

from collections import Counter, deque
from typing import Dict, List, Tuple

GESTURE_LABELS = {
    0: "Forward",
    1: "Back",
    2: "Up",
    3: "Down",
    4: "Left",
    5: "Right",
    6: "Stop",
    7: "Land",
}


class HandGestureDetector:
    """Detect hand gestures from MediaPipe hand landmarks for Aerosign.

    Gesture set:
      - Single index finger arrow direction: Up, Down, Left, Right
      - Two-finger (index + middle) up/down: Forward/Back
      - Two hand Open palm: Stop
      - Two hand Closed fist: Land
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self.hands.close()

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process a camera frame and return the annotated image with the detected gesture ID."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.hands.process(rgb_image)
        rgb_image.flags.writeable = True

        gesture_id = -1
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                landmark_lists = [self._calc_landmark_list(image, hand_landmarks) for hand_landmarks in results.multi_hand_landmarks]
                gesture_id = self._classify_two_hand_gesture(landmark_lists)
            else:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list = self._calc_landmark_list(image, hand_landmarks)
                gesture_id = self._classify_gesture(landmark_list)
                if gesture_id in {0, 1, 2, 3, 4, 5}:
                    self._draw_direction(image, landmark_list)
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                )
        return image, gesture_id

    def _calc_landmark_list(self, image: np.ndarray, landmarks) -> List[Tuple[int, int]]:
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_points: List[Tuple[int, int]] = []

        for landmark in landmarks.landmark:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_points.append((x, y))

        return landmark_points

    def _hand_scale(self, landmark_list: List[Tuple[int, int]]) -> float:
        wrist = np.array(landmark_list[0], dtype=float)
        middle_mcp = np.array(landmark_list[9], dtype=float)
        return max(np.linalg.norm(middle_mcp - wrist), 1.0)

    def _finger_states(self, landmark_list: List[Tuple[int, int]]) -> Dict[str, bool]:
        """Detect whether each finger is extended independent of pointing direction."""
        hand_scale = self._hand_scale(landmark_list)

        def is_finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip = np.array(landmark_list[tip_idx], dtype=float)
            pip = np.array(landmark_list[pip_idx], dtype=float)
            mcp = np.array(landmark_list[mcp_idx], dtype=float)
            dist_tip_pip = np.linalg.norm(tip - pip)
            dist_tip_mcp = np.linalg.norm(tip - mcp)
            return dist_tip_pip > hand_scale * 0.35 and dist_tip_mcp > hand_scale * 0.7

        return {
            "index": is_finger_extended(8, 6, 5),
            "middle": is_finger_extended(12, 10, 9),
            "ring": is_finger_extended(16, 14, 13),
            "pinky": is_finger_extended(20, 18, 17),
        }

    def _point_direction(self, landmark_list: List[Tuple[int, int]]) -> str:
        """Return the dominant direction of the index finger pointer."""
        hand_scale = self._hand_scale(landmark_list)
        tip = np.array(landmark_list[8], dtype=float)
        pip = np.array(landmark_list[6], dtype=float)

        # main() flips the live camera horizontally for a user-facing mirror view.
        # Compensate for that flip here so left/right gestures stay aligned with
        # the real-world motion the drone should follow.
        dx = -(tip[0] - pip[0])
        dy = tip[1] - pip[1]

        magnitude = np.hypot(dx, dy)
        if magnitude < hand_scale * 0.4:
            return "Center"

        if abs(dx) > abs(dy) * 0.7 and abs(dx) > hand_scale * 0.25:
            return "Right" if dx > 0 else "Left"
        if abs(dy) > abs(dx) * 0.7 and abs(dy) > hand_scale * 0.25:
            return "Down" if dy > 0 else "Up"

        return "Center"

    def _is_open_palm(self, landmark_list: List[Tuple[int, int]], states: Dict[str, bool]) -> bool:
        hand_size = self._hand_scale(landmark_list)
        if hand_size < 1:
            return False

        if sum(states[finger] for finger in ["index", "middle", "ring", "pinky"]) < 3:
            return False

        wrist = np.array(landmark_list[0], dtype=float)
        tip_indices = [4, 8, 12, 16, 20]
        tips = [np.array(landmark_list[i], dtype=float) for i in tip_indices]
        distances = [np.linalg.norm(tip - wrist) for tip in tips]

        if np.mean(distances[1:]) < hand_size * 0.65:
            return False

        separations = [
            np.linalg.norm(tips[0] - tips[1]),
            np.linalg.norm(tips[1] - tips[2]),
            np.linalg.norm(tips[2] - tips[3]),
            np.linalg.norm(tips[3] - tips[4]),
        ]
        if sum(1 for sep in separations if sep > hand_size * 0.2) < 3:
            return False

        return True

    def _is_closed_fist(self, landmark_list: List[Tuple[int, int]], states: Dict[str, bool]) -> bool:
        hand_size = self._hand_scale(landmark_list)
        if hand_size < 1:
            return False

        if sum(states[finger] for finger in ["index", "middle", "ring", "pinky"]) > 2:
            return False

        wrist = np.array(landmark_list[0], dtype=float)
        tip_indices = [4, 8, 12, 16, 20]
        tips = [np.array(landmark_list[i], dtype=float) for i in tip_indices]
        distances = [np.linalg.norm(tip - wrist) for tip in tips]
        if max(distances[1:]) > hand_size * 0.9:
            return False

        tip_pip_dists = [
            np.linalg.norm(np.array(landmark_list[8], dtype=float) - np.array(landmark_list[6], dtype=float)),
            np.linalg.norm(np.array(landmark_list[12], dtype=float) - np.array(landmark_list[10], dtype=float)),
            np.linalg.norm(np.array(landmark_list[16], dtype=float) - np.array(landmark_list[14], dtype=float)),
            np.linalg.norm(np.array(landmark_list[20], dtype=float) - np.array(landmark_list[18], dtype=float)),
        ]
        return max(tip_pip_dists) < hand_size * 0.6 or np.mean(tip_pip_dists) < hand_size * 0.55

    def _classify_gesture(self, landmark_list: List[Tuple[int, int]]) -> int:
        states = self._finger_states(landmark_list)

        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            direction = self._point_direction(landmark_list)
            if direction == "Up":
                return 0
            if direction == "Down":
                return 1
            return -1

        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            direction = self._point_direction(landmark_list)
            return {"Up": 2, "Down": 3, "Left": 4, "Right": 5}.get(direction, -1)

        return -1

    def _classify_two_hand_gesture(self, landmark_lists: List[List[Tuple[int, int]]]) -> int:
        left_states = self._finger_states(landmark_lists[0])
        right_states = self._finger_states(landmark_lists[1])

        if self._is_open_palm(landmark_lists[0], left_states) and self._is_open_palm(landmark_lists[1], right_states):
            return 6
        if self._is_closed_fist(landmark_lists[0], left_states) and self._is_closed_fist(landmark_lists[1], right_states):
            return 7

        return -1

    def _draw_direction(self, image: np.ndarray, landmark_list: List[Tuple[int, int]]) -> None:
        tip = tuple(landmark_list[8])
        pip = tuple(landmark_list[6])
        cv2.arrowedLine(image, pip, tip, (0, 255, 255), 3, tipLength=0.2)


class GestureBuffer:
    """Stabilize gesture detection by keeping a short history of results.

    The buffer returns a gesture only when one result appears consistently
    within the last few frames, reducing flicker from transient misreads.
    """

    def __init__(self, buffer_len: int = 10):
        self.buffer_len = buffer_len
        self._buffer = deque(maxlen=buffer_len)
        self._last_gesture = -1
        self._no_gesture_count = 0

    def add_gesture(self, gesture_id: int) -> None:
        self._buffer.append(gesture_id)
        if gesture_id == -1:
            self._no_gesture_count += 1
        else:
            self._no_gesture_count = 0

    def get_gesture(self) -> int:
        if not self._buffer:
            return self._last_gesture

        gesture_counts = Counter(self._buffer).most_common()
        if not gesture_counts:
            return self._last_gesture

        gesture_id, count = gesture_counts[0]
        if gesture_id == -1:
            if self._no_gesture_count >= max(2, int(self.buffer_len * 0.6)):
                self._last_gesture = -1
            return self._last_gesture

        if count >= max(2, int(self.buffer_len * 0.6)):
            self._buffer.clear()
            self._last_gesture = gesture_id
            return self._last_gesture

        return self._last_gesture

