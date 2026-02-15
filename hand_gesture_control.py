"""AI hand-gesture robot controller.

Uses MediaPipe Hands to detect finger states from a webcam feed and maps
recognized gestures to simple robot movement commands.

Commands are printed and can optionally be sent over a serial connection.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp

try:
    import serial  # type: ignore
except Exception:  # serial is optional
    serial = None


@dataclass
class RobotController:
    """Send commands to a robot over serial or stdout."""

    port: Optional[str]
    baudrate: int = 9600
    cooldown_seconds: float = 0.4

    def __post_init__(self) -> None:
        self._last_command = ""
        self._last_sent_at = 0.0
        self._serial_conn = None

        if self.port:
            if serial is None:
                raise RuntimeError(
                    "pyserial is not installed. Install it or run without --port."
                )
            self._serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # wait for microcontroller reset

    def send(self, command: str) -> None:
        now = time.time()
        if (
            command == self._last_command
            and now - self._last_sent_at < self.cooldown_seconds
        ):
            return

        payload = f"{command}\n"
        print(f"[ROBOT] {command}")
        if self._serial_conn:
            self._serial_conn.write(payload.encode("utf-8"))

        self._last_command = command
        self._last_sent_at = now

    def close(self) -> None:
        if self._serial_conn:
            self._serial_conn.close()


def is_finger_up(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """Return True when a finger is extended (tip above PIP in image coordinates)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def detect_gesture(landmarks) -> str:
    """Map landmark positions to a simple set of gesture commands."""
    thumb_up = landmarks[4].x > landmarks[3].x
    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)

    up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    if up_count == 0:
        return "STOP"
    if up_count == 5:
        return "FORWARD"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "RIGHT"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "LEFT"
    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "BACKWARD"
    return "IDLE"


def run(camera_index: int, controller: RobotController) -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            command = "IDLE"
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = hand.landmark
                command = detect_gesture(landmarks)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            controller.send(command)

            cv2.putText(
                frame,
                f"Gesture Command: {command}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "q: quit",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("AI Hand Gesture Robot Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI hand-gesture robot control")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--port", type=str, default=None, help="Serial port (optional)")
    parser.add_argument("--baudrate", type=int, default=9600, help="Serial baudrate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = RobotController(port=args.port, baudrate=args.baudrate)
    try:
        run(camera_index=args.camera, controller=controller)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
