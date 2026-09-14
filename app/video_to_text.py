from __future__ import annotations

import math
from pathlib import Path

from app.ollama_normalizer import ollama_normalizer
from app.recognizer import VideoDecodeError, recognizer


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

        if expected_signs > 0:
            segment_count = expected_signs
        else:
            segment_count = max(1, int(math.floor(duration / segment_seconds + 0.5)))
        segment_count = min(segment_count, 30)
        actual_segment_seconds = duration / segment_count

        segments = []
        labels = []
        for index in range(segment_count):
            start = index * actual_segment_seconds
            end = duration if index == segment_count - 1 else (index + 1) * actual_segment_seconds
            predictions = recognizer.predict_interval(path, start, end, top_k=3)
            best = predictions[0]
            accepted = best.confidence >= min_confidence
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
            raise NoSignsRecognized(
                "Không có đoạn nào đạt ngưỡng tin cậy; hãy giảm min_confidence hoặc chỉnh expected_signs"
            )
        text = ollama_normalizer.labels_to_text(labels)
        return {
            "text": text,
            "labels": labels,
            "segments": segments,
            "duration_seconds": round(duration, 3),
            "segment_seconds": round(actual_segment_seconds, 3),
            "recognition_model": recognizer.model_name,
            "language_model": ollama_normalizer.model,
        }


video_to_text_service = VideoToTextService()
