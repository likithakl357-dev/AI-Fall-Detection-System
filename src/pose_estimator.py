from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("models/pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

class PoseEstimator:
    def __init__(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists():
            print("Downloading MediaPipe Pose Landmarker model (first run only)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as exc:
                raise RuntimeError(
                    "Could not download the pose model. Check your internet connection "
                    f"and try again. Details: {exc}"
                )
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=3,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25,
            min_tracking_confidence=0.25,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp_ms += 40
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        if not result.pose_landmarks:
            return None
        # Select the pose with the largest visible torso/limb evidence.
        return max(result.pose_landmarks, key=self._pose_score)

    @staticmethod
    def _pose_score(pose):
        ids = [11, 12, 23, 24, 25, 26, 27, 28]
        return sum(1 for i in ids if pose[i].visibility > 0.2)

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
