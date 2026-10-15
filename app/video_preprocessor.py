from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ASSET_PATHS = [
    BASE_DIR / "models" / "hand_landmarker.task",
    BASE_DIR.parent / "HandSign" / "hand_landmarker.task",
]


def smart_crop_frame(frame: np.ndarray, focus_x: float = 0.5) -> np.ndarray:
    """Crop widescreen frames to a square centered near focus_x to prevent stretching."""
    h, w = frame.shape[:2]
    if w <= h:
        return frame
    target_size = h
    center_x = int(focus_x * w)
    start_x = max(0, min(w - target_size, center_x - target_size // 2))
    return frame[:, start_x : start_x + target_size]


class HandActivityDetector:
    """Detects active hand gestures to trim idle padding and locate the hand region."""

    def __init__(self) -> None:
        self._detector: Any = None
        self._lock = threading.Lock()
        self._available: bool | None = None

    def _get_detector(self) -> Any:
        if self._available is False:
            return None
        if self._detector is not None:
            return self._detector
        with self._lock:
            if self._detector is not None:
                return self._detector
            try:
                import mediapipe as mp

                task_file = None
                for path in MODEL_ASSET_PATHS:
                    if path.is_file():
                        task_file = path
                        break

                if not task_file:
                    self._available = False
                    return None

                base_options = mp.tasks.BaseOptions(model_asset_path=str(task_file))
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.25,
                )
                self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
                self._available = True
                return self._detector
            except Exception:
                self._available = False
                return None

    def analyze_hands(self, video_path: Path | str, sample_step: int = 2) -> dict[str, Any]:
        """Analyze video frames for hand activity, bounding box center, and active interval."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return {
                "has_hands": True,
                "overall_start": 0.0,
                "overall_end": 0.0,
                "hand_center_x": 0.5,
                "active_frames": 0,
                "total_frames": 0,
            }

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        duration = total / fps if fps > 0 else 0.0

        detector = self._get_detector()
        if detector is None or total < 2:
            capture.release()
            return {
                "has_hands": True,
                "overall_start": 0.0,
                "overall_end": duration,
                "hand_center_x": 0.5,
                "active_frames": total,
                "total_frames": total,
            }

        active_frame_indices: list[int] = []
        hand_x_positions: list[float] = []

        try:
            import mediapipe as mp

            for idx in range(0, total, max(1, sample_step)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)
                if res.hand_landmarks:
                    active_frame_indices.append(idx)
                    for hand in res.hand_landmarks:
                        for pt in hand:
                            hand_x_positions.append(pt.x)
        except Exception:
            # Fall back safely if processing a frame errors
            pass
        finally:
            capture.release()

        if not active_frame_indices:
            return {
                "has_hands": False,
                "overall_start": 0.0,
                "overall_end": duration,
                "hand_center_x": 0.5,
                "active_frames": 0,
                "total_frames": total,
            }

        # Calculate active time range with slight padding so gesture start/end are kept
        start_frame = max(0, active_frame_indices[0] - sample_step)
        end_frame = min(total - 1, active_frame_indices[-1] + sample_step)

        start_seconds = start_frame / fps if fps > 0 else 0.0
        end_seconds = end_frame / fps if fps > 0 else duration

        # Ensure valid range
        if end_seconds - start_seconds < 0.5:
            # If active duration is very short, retain original duration
            start_seconds = 0.0
            end_seconds = duration

        avg_x = (
            float(sum(hand_x_positions) / len(hand_x_positions))
            if hand_x_positions
            else 0.5
        )

        return {
            "has_hands": True,
            "overall_start": round(start_seconds, 3),
            "overall_end": round(end_seconds, 3),
            "hand_center_x": round(avg_x, 3),
            "active_frames": len(active_frame_indices),
            "total_frames": total,
        }


hand_activity_detector = HandActivityDetector()
