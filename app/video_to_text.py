from __future__ import annotations

import math
from pathlib import Path

from app.landmark_verifier import landmark_verifier
from app.ollama_normalizer import OllamaCloudError, ollama_normalizer
from app.recognizer import VideoDecodeError, recognizer
from app.video_preprocessor import hand_activity_detector


class NoSignsRecognized(RuntimeError):
    pass


def video_duration(path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise VideoDecodeError("Thiếu OpenCV để đọc video") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError("Không mở được video")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or frames < 2:
        raise VideoDecodeError("Video không có đủ khung hình hợp lệ")
    return frames / fps


class VideoToTextService:
    def translate(
        self,
        path: Path,
        segment_seconds: float = 4.0,
        expected_signs: int = 0,
        min_confidence: float = 0.35,
    ) -> dict:
        duration = video_duration(path)
        if duration > 120:
            raise VideoDecodeError("Video tối đa 120 giây")

        analysis = hand_activity_detector.analyze_hands(path)
        if not analysis.get("has_hands", True):
            raise NoSignsRecognized(
                "Chưa phát hiện thấy bàn tay trong video. Vui lòng quay rõ nửa thân trên và hai bàn tay."
            )

        active_start = float(analysis.get("overall_start", 0.0))
        active_end = float(analysis.get("overall_end", duration))
        focus_x = float(analysis.get("hand_center_x", 0.5))

        if active_end > active_start + 0.3:
            active_duration = active_end - active_start
        else:
            active_start = 0.0
            active_end = duration
            active_duration = duration

        if expected_signs > 0:
            segment_count = expected_signs
        else:
            segment_count = max(1, int(math.floor(active_duration / segment_seconds + 0.5)))
        segment_count = min(segment_count, 30)
        actual_segment_seconds = active_duration / segment_count

        segments = []
        labels = []
        best_unaccepted_label: str | None = None
        best_unaccepted_confidence: float = 0.0

        for index in range(segment_count):
            start = active_start + index * actual_segment_seconds
            end = active_end if index == segment_count - 1 else active_start + (index + 1) * actual_segment_seconds
            predictions = recognizer.predict_interval(
                path, start, end, top_k=3, focus_x=focus_x
            )
            best = predictions[0]
            accepted = best.confidence >= min_confidence
            if not accepted and best.confidence > best_unaccepted_confidence:
                best_unaccepted_confidence = best.confidence
                best_unaccepted_label = best.label

            segments.append(
                {
                    "index": index,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "label": best.label,
                    "confidence": best.confidence,
                    "accepted": accepted,
                    "top_predictions": [
                        {"label": item.label, "confidence": item.confidence}
                        for item in predictions
                    ],
                }
            )
            if accepted and (not labels or labels[-1] != best.label):
                labels.append(best.label)

        if not labels:
            verified = landmark_verifier.verify(
                path,
                candidate_labels=[best_unaccepted_label] if best_unaccepted_label else None,
            )
            if verified:
                v_label, v_conf = verified
                labels.append(v_label)
                if segments:
                    segments[0]["label"] = v_label
                    segments[0]["confidence"] = v_conf
                    segments[0]["accepted"] = True
                    segments[0]["top_predictions"].insert(
                        0, {"label": v_label, "confidence": v_conf}
                    )

        if not labels:
            tip = ""
            if best_unaccepted_label:
                tip = f" (gần giống '{best_unaccepted_label}' với độ tin cậy {best_unaccepted_confidence * 100:.1f}%)"
            raise NoSignsRecognized(
                f"Không có cử chỉ nào đạt ngưỡng tin cậy {min_confidence * 100:.0f}%{tip}. "
                "Hãy hạ ngưỡng tin cậy trên thanh trượt hoặc thực hiện cử chỉ thuộc 100 từ vựng hỗ trợ."
            )
        try:
            text = ollama_normalizer.labels_to_text(labels)
            language_model = ollama_normalizer.model
        except OllamaCloudError:
            text = ", ".join(labels)
            language_model = "raw-labels-fallback"
        return {
            "text": text,
            "labels": labels,
            "segments": segments,
            "duration_seconds": round(duration, 3),
            "segment_seconds": round(actual_segment_seconds, 3),
            "recognition_model": recognizer.model_name,
            "language_model": language_model,
        }


video_to_text_service = VideoToTextService()
