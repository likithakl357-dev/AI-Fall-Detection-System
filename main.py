import argparse
import os
import cv2
from pathlib import Path
from src.pose_estimator import PoseEstimator
from src.detector import FallDetector

def main():
    parser = argparse.ArgumentParser(description="AI Fall Detection V2")
    parser.add_argument("--source", default="data/videos/fall_test.mp4",
                        help="Video path or camera index (use 0 for webcam)")
    parser.add_argument("--output", default="", help="Optional output video path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    output_path = args.output
    writer = None
    if not output_path and isinstance(source, str):
        output_path = str(Path("outputs") / f"{Path(source).stem}_v2_annotated.mp4")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    print("Starting Fall Detection V2. Press Q/Esc to quit, P to pause.")
    print("Note: This is an educational prototype, not a certified safety system.")
    detector = FallDetector(fps=fps)

    try:
        with PoseEstimator() as pose_estimator:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                landmarks = pose_estimator.detect(frame)
                result = detector.update(landmarks, frame.shape[1], frame.shape[0])
                display = detector.draw(frame, landmarks, result)

                cv2.imshow("AI Fall Detection - Version 2", display)
                if writer:
                    writer.write(display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    while True:
                        k = cv2.waitKey(0) & 0xFF
                        if k in (ord("p"), ord("q"), 27):
                            break
                    if k in (ord("q"), 27):
                        break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if output_path:
        print(f"Annotated video saved to: {output_path}")

if __name__ == "__main__":
    main()
