# AI Fall Detection System — Version 2

A lightweight educational fall-detection prototype using OpenCV and MediaPipe Pose Landmarker.

## What's improved in V2

- Does not require face, nose, head, or feet landmarks to estimate posture.
- Uses visible shoulders, hips and available leg landmarks.
- Lowers pose detection/tracking thresholds to improve recovery when part of the body is clipped.
- Separates `TRACKING LOST`, `POSE UNCLEAR`, and `NO PERSON`.
- A missing pose is not immediately treated as an empty frame.
- Latches a fall warning for 5 seconds after a rapid-descent + lying transition.
- Saves an annotated output video.

**Important limitation:** A pose model cannot guarantee detection if too little of the person is visible, if the person is heavily occluded, or if the person is outside the camera view. When evidence is insufficient, this version tries to show an uncertain/tracking state rather than confidently claiming there is no person. A fall outside the camera view cannot be confirmed by this camera alone.

This is an educational prototype, not a certified medical, security, or emergency-response system. It can produce false positives and false negatives.

## 1. Requirements

- Windows 10/11, 64-bit
- Python 3.11 recommended (Python 3.12 may also work depending on package wheels)
- VS Code
- Internet connection on first run to download the MediaPipe model

## 2. Setup (PowerShell)

Open this folder in VS Code, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed but Python 3.12 is available, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Add videos

Put MP4 videos in `data/videos/`. The default video name is `fall_test.mp4`.

```text
data/
  videos/
    fall_test.mp4
    sitting_test.mp4
    side_fall.mp4
```

## 4. Run

Default video:

```powershell
python main.py
```

Choose another video:

```powershell
python main.py --source "data/videos/sitting_test.mp4"
```

Use webcam:

```powershell
python main.py --source 0
```

Choose output path:

```powershell
python main.py --source "data/videos/side_fall.mp4" --output "outputs/side_fall_result.mp4"
```

Keys: `Q` or `Esc` = quit; `P` = pause/resume.

## 5. Output

Annotated videos are saved in `outputs/`. The first run downloads the lightweight Pose Landmarker model into `models/`.

## 6. Testing advice

Test one video at a time. Use clips with a fixed camera, adequate lighting, and as much of the person's torso and hips in frame as possible. Test standing, sitting, walking, deliberate lying down, and staged falls separately. Do not deliberately perform a real fall to test the software.
