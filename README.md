# Hand Gesture Control Robot

This project now includes an **AI-based hand gesture controller** that uses a webcam and MediaPipe hand tracking to issue robot commands.

## Features
- Real-time hand landmark detection using MediaPipe.
- Gesture-to-command mapping for:
  - `FORWARD` (open palm)
  - `BACKWARD` (thumb + pinky)
  - `LEFT` (index + middle)
  - `RIGHT` (index only)
  - `STOP` (fist)
  - `IDLE` (unrecognized state)
- Optional serial output to Arduino/robot controller.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Without serial output:
```bash
python3 hand_gesture_control.py
```

With serial output (example):
```bash
python3 hand_gesture_control.py --port /dev/ttyUSB0 --baudrate 9600
```

## Controls
- Show gesture in front of webcam.
- Press `q` to quit.

## Notes
- If handedness causes mirrored behavior for thumb detection, invert the thumb logic in `detect_gesture`.
- Make sure your microcontroller code expects newline-terminated command strings.
