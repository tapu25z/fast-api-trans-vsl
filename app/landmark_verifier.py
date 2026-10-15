from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
HANDSIGN_DIR = BASE_DIR.parent / "HandSign"
if str(HANDSIGN_DIR) not in sys.path:
    sys.path.insert(0, str(HANDSIGN_DIR))


class LandmarkVerifier:
    """Verifies and disambiguates signs using MediaPipe landmarks and HandSign DTW."""

    def __init__(self) -> None:
        self.handsign_dir = HANDSIGN_DIR
        self._evaluator: Any = None
        self._detector: Any = None

    def get_evaluator(self) -> Any:
        if self._evaluator is None:
            from sign_eval.evaluator import DEFAULT_CALIBRATION_PATH, SignEvaluator

            self._evaluator = SignEvaluator(DEFAULT_CALIBRATION_PATH)
        return self._evaluator

    def _extract_landmarks(self, video_path: Path | str) -> tuple[np.ndarray, np.ndarray] | None:
        import mediapipe as mp

        base_options = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        hand_task = self.handsign_dir / "hand_landmarker.task"
        face_task = self.handsign_dir / "face_landmarker.task"
        if not hand_task.is_file() or not face_task.is_file():
            return None

        hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options(model_asset_path=str(hand_task)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )
        )
        face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_options(model_asset_path=str(face_task)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
        )

        cap = cv2.VideoCapture(str(video_path))
        all_landmarks = []
        all_validity = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            h_res = hand_detector.detect(mp_image)
            f_res = face_detector.detect(mp_image)
            pts = np.zeros((48, 2), dtype=np.float32)
            val = np.zeros(3, dtype=bool)

            for i, h in enumerate(h_res.hand_landmarks):
                if i >= 2:
                    break
                slot = 0 if h_res.handedness[i][0].category_name == "Left" else 1
                val[slot] = True
                offset = 0 if slot == 0 else 21
                for j, lm in enumerate(h):
                    pts[offset + j] = (lm.x, lm.y)

            if f_res.face_landmarks:
                val[2] = True
                f_lms = f_res.face_landmarks[0]
                for j, f_idx in enumerate([33, 263, 1, 13, 234, 454]):
                    pts[42 + j] = (f_lms[f_idx].x, f_lms[f_idx].y)

            all_landmarks.append(pts)
            all_validity.append(val)
        cap.release()

        landmarks = np.array(all_landmarks, dtype=np.float32)
        validity = np.array(all_validity, dtype=bool)
        if len(landmarks) < 2 or not validity[:, :2].any():
            return None
        return landmarks, validity

    def verify(
        self,
        video_path: Path | str,
        candidate_labels: list[str] | None = None,
    ) -> tuple[str, float] | None:
        """Inspect video landmarks with DTW to recognize gestures robustly against background noise."""
        try:
            from sign_eval.landmarks import trim_to_active_hands, zscore_normalize_sequence

            data = self._extract_landmarks(video_path)
            if not data:
                return None
            landmarks, validity = data

            trimmed_lms, trimmed_val = trim_to_active_hands(landmarks, validity)
            normalized_lms = zscore_normalize_sequence(trimmed_lms)

            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                np.savez(tmp.name, landmarks=normalized_lms, validity=trimmed_val)
                tmp_path = tmp.name

            # Check if right hand fingertips are tapping near left wrist (Pulse / Bắt mạch / Bệnh nhân)
            wrist_pulse_touch = False
            for f in range(len(trimmed_lms)):
                if trimmed_val[f, 0] and trimmed_val[f, 1]:
                    lw = trimmed_lms[f, 0]  # left wrist joint
                    rf_index = trimmed_lms[f, 21 + 8]  # right index tip
                    rf_middle = trimmed_lms[f, 21 + 12]  # right middle tip
                    if np.linalg.norm(lw - rf_index) < 0.16 or np.linalg.norm(lw - rf_middle) < 0.16:
                        wrist_pulse_touch = True
                        break

            evaluator = self.get_evaluator()

            # If pulse-touching gesture is detected, evaluate Bệnh nhân
            if wrist_pulse_touch:
                res_bn = evaluator.evaluate(
                    tmp_path, "Bệnh nhân", recognition={"predicted_label": "Bệnh nhân", "confidence": 0.9}
                ).to_dict()
                sc = res_bn.get("form", {}).get("score")
                if sc and sc >= 75.0:
                    return "Bệnh nhân", round(sc / 100.0, 4)

            # Otherwise test against top candidate labels
            if candidate_labels:
                best_label = None
                best_score = 0.0
                for label in candidate_labels[:5]:
                    if label in evaluator.available_labels():
                        try:
                            r = evaluator.evaluate(
                                tmp_path, label, recognition={"predicted_label": label, "confidence": 0.9}
                            ).to_dict()
                            sc = r.get("form", {}).get("score")
                            if sc and sc >= 80.0 and sc > best_score:
                                best_score = sc
                                best_label = label
                        except Exception:
                            continue
                if best_label:
                    return best_label, round(best_score / 100.0, 4)

            return None
        except Exception:
            return None


landmark_verifier = LandmarkVerifier()
