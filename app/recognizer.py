from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path


MODEL_NAME = "star092304/vi-sign-language-videomae-base"


class RecognizerUnavailable(RuntimeError):
    """Raised when optional ML dependencies or model weights are unavailable."""


class VideoDecodeError(ValueError):
    """Raised when an uploaded file cannot be decoded as a video."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


# A few labels in the public checkpoint were saved with broken legacy encoding.
LABEL_CORRECTIONS = {
    "Ban ngÖy": "Ban ngày",
    "BÖn tay": "Bàn tay",
    "Ch£ng ta": "Chúng ta",
    "ChÉn": "Chán",
    "ChÖo": "Chào",
    "C†": "Cá",
    "C†ch ly": "Cách ly",
    "GhÇt": "Ghét",
    "Gi£p": "Giúp",
    "Hìm nay": "Hôm nay",
    "Khai b†o": "Khai báo",
    "Khu c†ch ly": "Khu cách ly",
    "Kh¢c": "Khóc",
    "Ng¢n tay": "Ngón tay",
    "NhÖ": "Nhà",
    "Nhçn": "Nhận",
    "N¢i": "Nơi",
    "Nìn ¢i": "Nôn ói",
    "Ph°a sau": "Phía sau",
    "Th°ch": "Thích",
    "Tìi": "Tôi",
    "Xe m†y": "Xe máy",
    "Xin phÇp": "Xin phép",
}


def clean_label(label: str) -> str:
    return LABEL_CORRECTIONS.get(label, label)


class VideoMAEVSLRecognizer:
    """Lazy-loaded isolated-word Vietnamese sign-language recognizer."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "not-loaded"
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def status(self) -> dict[str, str | bool | None]:
        return {
            "model": self.model_name,
            "loaded": self._model is not None,
            "device": self._device,
            "load_error": self._load_error,
            "task": "isolated-word-video-classification",
        }

    def _select_device(self, torch) -> str:
        requested = os.getenv("VSL_DEVICE", "auto").lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                import torch.nn as nn
                from huggingface_hub import hf_hub_download
                from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

                processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
                model = VideoMAEForVideoClassification.from_pretrained(
                    self.model_name,
                    ignore_mismatched_sizes=True,
                )

                # The published model used a sequential classification head.
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Dropout(0.3),
                    nn.Linear(in_features, model.config.num_labels),
                )
                checkpoint = hf_hub_download(
                    repo_id=self.model_name,
                    filename="classifier_sequential.pth",
                )
                state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)

                device = self._select_device(torch)
                model = model.to(device)
                model.eval()
                self._processor = processor
                self._model = model
                self._torch = torch
                self._device = device
                self._load_error = None
            except Exception as exc:  # dependency/download errors become a clear API response
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise RecognizerUnavailable(self._load_error) from exc

    @staticmethod
    def _load_frames(
        video_path: Path,
        frame_count: int = 16,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> list:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RecognizerUnavailable(
                "Thiếu thư viện OpenCV. Hãy cài requirements-ml.txt."
            ) from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoDecodeError("Không mở được video")

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames = []
        if total > 0:
            start_frame = 0
            end_frame = max(total - 1, 0)
            if fps > 0 and start_seconds is not None:
                start_frame = min(max(int(start_seconds * fps), 0), end_frame)
            if fps > 0 and end_seconds is not None:
                end_frame = min(max(int(end_seconds * fps) - 1, start_frame), end_frame)
            indices = np.linspace(start_frame, end_frame, frame_count).astype(int)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()

        if len(frames) < 2:
            raise VideoDecodeError("Video không có đủ khung hình hợp lệ")
        while len(frames) < frame_count:
            frames.append(frames[-1].copy())
        return frames[:frame_count]

    def predict_interval(
        self,
        video_path: Path,
        start_seconds: float | None,
        end_seconds: float | None,
        top_k: int = 5,
    ) -> list[Prediction]:
        self.load()
        frames = self._load_frames(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        inputs = self._processor(frames, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}

        with self._torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1)
            values, indices = self._torch.topk(probabilities, k=min(top_k, len(probabilities)))

        return [
            Prediction(
                label=clean_label(self._model.config.id2label[int(index)]),
                confidence=round(float(value), 6),
            )
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        ]


recognizer = VideoMAEVSLRecognizer()
