from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path


DEFAULT_MODEL = "openai/whisper-small"


class SpeechRecognizerUnavailable(RuntimeError):
    pass


class AudioDecodeError(RuntimeError):
    pass


class VietnameseSpeechRecognizer:
    def __init__(self) -> None:
        self.model_name = os.getenv("SPEECH_MODEL", DEFAULT_MODEL)
        self.requested_device = os.getenv("SPEECH_DEVICE", "auto").lower()
        self._processor = None
        self._model = None
        self._device = "not-loaded"
        self._lock = threading.Lock()

    @property
    def status(self) -> dict:
        return {
            "model": self.model_name,
            "loaded": self._model is not None,
            "device": self._device,
        }

    def _select_device(self, torch) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            device = self._select_device(torch)
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                use_safetensors=True,
            ).to(device)
            model.eval()
        except Exception as exc:
            raise SpeechRecognizerUnavailable(f"Không tải được model giọng nói: {exc}") from exc
        self._processor = processor
        self._model = model
        self._device = device

    @staticmethod
    def _decode_audio(path: Path):
        try:
            import imageio_ffmpeg
            import numpy as np

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            process = subprocess.run(
                [
                    ffmpeg,
                    "-v", "error",
                    "-i", str(path),
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    "-f", "f32le",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=180,
            )
            if process.returncode != 0:
                detail = process.stderr.decode("utf-8", errors="replace")[-800:]
                raise AudioDecodeError(f"Không đọc được âm thanh: {detail}")
            audio = np.frombuffer(process.stdout, dtype=np.float32).copy()
            if audio.size < 1600:
                raise AudioDecodeError("Âm thanh quá ngắn hoặc không có tiếng")
            return audio
        except ImportError as exc:
            raise SpeechRecognizerUnavailable("Thiếu imageio-ffmpeg hoặc numpy") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError("Âm thanh xử lý quá lâu") from exc

    def transcribe(self, path: Path) -> str:
        audio = self._decode_audio(path)
        with self._lock:
            self._load()
            try:
                import torch

                inputs = self._processor(
                    audio,
                    sampling_rate=16_000,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = inputs.input_features.to(self._device)
                attention_mask = inputs.attention_mask.to(self._device)
                with torch.inference_mode():
                    predicted_ids = self._model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        language="vi",
                        task="transcribe",
                    )
                text = self._processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True,
                )[0].strip()
            except Exception as exc:
                raise SpeechRecognizerUnavailable(f"Không nhận dạng được giọng nói: {exc}") from exc
        if not text:
            raise AudioDecodeError("Không nhận ra lời nói trong file")
        return text


speech_recognizer = VietnameseSpeechRecognizer()
