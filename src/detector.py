import math
from collections import deque
import cv2

# MediaPipe Pose landmark indices
LS, RS = 11, 12
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28

def midpoint(a, b):
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

def angle_at(a, b, c):
    # Angle ABC in degrees
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denom = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denom < 1e-6:
        return None
    dot = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / denom))
    return math.degrees(math.acos(dot))

class FallDetector:
    def __init__(self, fps=25):
        self.fps = fps
        self.history = deque(maxlen=max(8, int(fps * 2.5)))
        self.last_pose_time = None
        self.frame_count = 0
        self.fall_until = -1
        self.previous_state = "POSE UNCLEAR"
        self.last_center_y = None
        self.last_time = 0.0
        self.frame_seconds = 1.0 / max(1.0, fps)

    def _visible(self, p, threshold=0.20):
        return getattr(p, "visibility", 0.0) >= threshold

    def _extract(self, landmarks):
        if landmarks is None:
            return None
        # Do not require nose, eyes, ears, or feet. A clipped head is acceptable
        # when enough torso/hip/leg landmarks remain visible.
        needed = [LS, RS, LH, RH]
        visible_core = sum(self._visible(landmarks[i]) for i in needed)
        if visible_core < 2:
            # Allow a partial torso plus visible knees/ankles to be tracked.
            torso_or_legs = [LS, RS, LH, RH, LK, RK, LA, RA]
            if sum(self._visible(landmarks[i], 0.25) for i in torso_or_legs) < 3:
                return None

        def point(i):
            p = landmarks[i]
            return (p.x, p.y) if self._visible(p, 0.15) else None

        shoulders = [point(LS), point(RS)]
        hips = [point(LH), point(RH)]
        knees = [point(LK), point(RK)]
        ankles = [point(LA), point(RA)]

        def avg(points):
            pts = [p for p in points if p is not None]
            if not pts:
                return None
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))

        shoulder_mid, hip_mid = avg(shoulders), avg(hips)
        knee_mid, ankle_mid = avg(knees), avg(ankles)
        if shoulder_mid is None and hip_mid is None:
            return None

        # Body axis uses shoulders-to-hips. No head landmark is used.
        torso_angle = None
        if shoulder_mid and hip_mid:
            dx = abs(shoulder_mid[0] - hip_mid[0])
            dy = abs(shoulder_mid[1] - hip_mid[1])
            torso_angle = math.degrees(math.atan2(dx, max(dy, 1e-5)))

        knee_angles = []
        for side in ((LH, LK, LA), (RH, RK, RA)):
            if all(point(i) is not None for i in side):
                a = angle_at(point(side[0]), point(side[1]), point(side[2]))
                if a is not None:
                    knee_angles.append(a)

        center = hip_mid or shoulder_mid
        return {
            "center": center,
            "torso_angle": torso_angle,
            "knee_angle": sum(knee_angles) / len(knee_angles) if knee_angles else None,
            "shoulder": shoulder_mid,
            "hip": hip_mid,
            "visible_count": sum(self._visible(p, 0.15) for p in landmarks),
            "partial": visible_core < 4,
        }

    def update(self, landmarks, width, height):
        self.frame_count += 1
        now = self.frame_count * self.frame_seconds

        pose = self._extract(landmarks)
        if pose is None:
            # A missing pose is not proof that the frame is empty. Keep a
            # separate uncertain state, especially after a recent detection.
            if self.last_pose_time is not None and now - self.last_pose_time <= 3.0:
                state = "TRACKING LOST"
            elif self.last_pose_time is None:
                state = "POSE UNCLEAR"
            else:
                state = "NO PERSON"
            if now <= self.fall_until:
                state = "FALL DETECTED"
            self.previous_state = state
            return {"state": state, "detail": "Pose not confidently visible", "partial": False}

        self.last_pose_time = now
        center_y = pose["center"][1] if pose["center"] else None
        velocity = 0.0
        if center_y is not None and self.last_center_y is not None:
            velocity = (center_y - self.last_center_y) / self.frame_seconds
        if center_y is not None:
            self.last_center_y = center_y

        torso_angle = pose["torso_angle"]
        knee = pose["knee_angle"]
        lying = torso_angle is not None and torso_angle >= 55

        if lying:
            state = "LYING"
        elif torso_angle is not None and torso_angle < 35:
            if knee is not None and knee < 145:
                state = "SITTING"
            else:
                state = "STANDING"
        elif torso_angle is not None:
            state = "POSE UNCLEAR"
        else:
            state = "POSE UNCLEAR"

        # Fall transition: rapid downward hip/torso-center movement followed
        # shortly by a horizontal torso. Keep the alert latched for 5 seconds.
        rapid_descent = velocity > 0.65
        self.history.append((now, state, center_y, velocity))
        recent_descent = any(t >= now - 1.2 and v > 0.65
                             for t, _, _, v in self.history)
        if lying and (rapid_descent or recent_descent):
            self.fall_until = now + 5.0

        if now <= self.fall_until:
            state = "FALL DETECTED"

        detail = "Partial body landmarks used" if pose["partial"] else ""
        if state == "LYING":
            detail = "Horizontal body posture; fall not confirmed"
        elif state == "FALL DETECTED":
            detail = "Rapid descent followed by lying posture"
        self.previous_state = state
        return {"state": state, "detail": detail, "partial": pose["partial"]}

    def draw(self, frame, landmarks, result):
        state = result["state"]
        colors = {
            "FALL DETECTED": (0, 0, 255),
            "LYING": (0, 210, 255),
            "STANDING": (0, 200, 0),
            "SITTING": (255, 180, 0),
            "WALKING": (255, 180, 0),
            "TRACKING LOST": (0, 140, 255),
            "POSE UNCLEAR": (180, 180, 180),
            "NO PERSON": (0, 180, 0),
        }
        color = colors.get(state, (180, 180, 180))
        if landmarks:
            # Draw visible body landmarks and only lines whose endpoints exist.
            connections = [
                (11, 12), (11, 23), (12, 24), (23, 24),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (23, 25), (25, 27), (24, 26), (26, 28)
            ]
            h, w = frame.shape[:2]
            pts = {}
            for i, p in enumerate(landmarks):
                if p.visibility > 0.2:
                    x, y = int(p.x * w), int(p.y * h)
                    pts[i] = (x, y)
                    cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)
            for a, b in connections:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], (255, 255, 0), 2)

        cv2.rectangle(frame, (10, 10), (min(frame.shape[1]-10, 500), 85), (25, 25, 25), -1)
        cv2.putText(frame, state, (22, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
        detail = result.get("detail", "")
        if detail:
            cv2.putText(frame, detail[:58], (22, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (230, 230, 230), 1, cv2.LINE_AA)
        return frame
