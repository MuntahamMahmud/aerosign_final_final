# Aerosign Gesture

Hand gesture detection for Aerosign-style drone commands using MediaPipe and OpenCV.

## Environment

- Python 3.11.9
- MediaPipe 0.10.14
- NumPy 2.4.6
- OpenCV (opencv-python) 5.0.0.93

## Files included

- `main.py` - camera capture, live gesture detection, and display.
- `gesture_control.py` - gesture classification using MediaPipe hand landmarks.
- `requirements` - required Python packages.
- `.gitignore` - excludes virtual environment files and temporary artifacts.

## Why `gesture_env` is excluded

The `gesture_env` folder is a local Python virtual environment. It is not tracked in Git because:

- it is machine-specific
- it contains installed packages and large binary files
- it can be recreated from `requirements`

## Setup

1. Create a virtual environment:

```powershell
python -m venv gesture_env
```

2. Activate it:

```powershell
.\gesture_env\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements
```

## Run

```powershell
python main.py
```

## Build a clickable Windows EXE

On Windows, double-click `build_exe.bat`. It uses the project's
`gesture_env`, installs PyInstaller there if needed, and creates:

```text
dist\AerosignGesture.exe
```

The EXE includes the gesture-control code, Python dependencies, and the
bundled Vosk model. Copy `dist\AerosignGesture.exe` to a Windows PC with a
camera and microphone; Python does not need to be installed on that PC.
The application writes runtime logs beside the EXE in `logs\aerosign.log`.
The camera and microphone drivers still need to be available on the target PC.

This script now defaults to a MAVLink UDP listener on
`udp:0.0.0.0:14550` at `57600` baud. Enable MAVLink from the video-feed button, then click
`Takeoff` to start the arm, GUIDED, and five-second takeoff sequence.

### Voice control

Switch the OpenCV control mode from `Gesture` to `Voice` to enable the microphone.
Voice mode uses the offline Vosk model at `models\vosk` and accepts only:

- `go` (forward)
- `back`
- `up`
- `down`

Minor pronunciation variations such as `goo`, `bak`, `op`, and `dawn` are
accepted, while unrelated words are ignored. Each accepted command sends one
velocity segment and then stops the vehicle, targeting approximately `0.5 m`
of movement. Install the voice dependencies from `requirements` and place a
Vosk model at `models\vosk` before enabling Voice mode.

### Override the default MAVLink transport

```powershell
python main.py --mavlink COM3 --baud 57600
```

Or use a different UDP endpoint:

```powershell
python main.py --mavlink udp:0.0.0.0:14550
```

To listen on a different UDP port, use another bind endpoint, for example:

```powershell
python main.py --mavlink udp:0.0.0.0:14551
```

## Notes

If you want to reuse this repository on another machine, recreate the virtual environment and install from `requirements` rather than copying `gesture_env`. If you add new dependencies, update `requirements` accordingly.
