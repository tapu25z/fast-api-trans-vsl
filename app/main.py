from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.ollama_normalizer import OllamaCloudError
from app.recognizer import RecognizerUnavailable, VideoDecodeError, recognizer
from app.speech_recognizer import AudioDecodeError, SpeechRecognizerUnavailable, speech_recognizer
from app.text_to_vsl_video import (
    DatasetUnavailable, UnknownLearningSign, VideoAssemblyError, text_to_vsl_video_service,
)
from app.video_to_text import NoSignsRecognized, video_to_text_service


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".webm", ".mp4"}


class VideoPrediction(BaseModel):
    label: str
    confidence: float


class VideoTextSegment(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    label: str
    confidence: float
    accepted: bool
    top_predictions: list[VideoPrediction]


class VideoToTextResponse(BaseModel):
    filename: str
    text: str
    labels: list[str]
    segments: list[VideoTextSegment]
    duration_seconds: float
    segment_seconds: float
    recognition_model: str
    language_model: str


class TextToVSLVideoRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1_000)
    exact: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text không được để trống")
        return value


class VSLVideoClip(BaseModel):
    label: str
    filename: str


class TextToVSLVideoResponse(BaseModel):
    normalized_text: str
    labels: list[str]
    unknown_words: list[str]
    model: str
    clips: list[VSLVideoClip]
    video_url: str


class SpeechToVSLVideoResponse(TextToVSLVideoResponse):
    transcript: str
    speech_model: str


async def read_upload(
    file: UploadFile,
    allowed_suffixes: set[str],
    max_size_mb: int,
    default_name: str,
) -> tuple[str, bytes]:
    suffix = Path(file.filename or default_name).suffix.lower()
    if suffix not in allowed_suffixes:
        supported = ", ".join(sorted(item.removeprefix(".").upper() for item in allowed_suffixes))
        raise HTTPException(status_code=415, detail=f"Chỉ hỗ trợ {supported}")
    limit = max_size_mb * 1024 * 1024
    content = await file.read(limit + 1)
    await file.close()
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"File tối đa {max_size_mb} MB")
    if not content:
        raise HTTPException(status_code=422, detail="File rỗng")
    return suffix, content


app = FastAPI(
    title="SignMate VSL API",
    description="Speech/Text → VSL và Video VSL → Text.",
    version="0.4.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/api/v1/models/vsl/status", tags=["models"])
def vsl_model_status() -> dict:
    return recognizer.status


@app.get("/api/v1/models/speech/status", tags=["models"])
def speech_model_status() -> dict:
    return speech_recognizer.status


@app.post(
    "/api/v1/translate/text-to-vsl-video",
    response_model=TextToVSLVideoResponse,
    tags=["speech-or-text-to-vsl"],
    summary="Chuẩn hóa text và ghép video VSL",
)
def translate_text_to_vsl_video(payload: TextToVSLVideoRequest) -> TextToVSLVideoResponse:
    try:
        if payload.exact:
            return TextToVSLVideoResponse(**text_to_vsl_video_service.translate_exact(payload.text))
        return TextToVSLVideoResponse(**text_to_vsl_video_service.translate(payload.text))
    except UnknownLearningSign as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaCloudError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DatasetUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VideoAssemblyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/api/v1/translate/speech-to-vsl-video",
    response_model=SpeechToVSLVideoResponse,
    tags=["speech-or-text-to-vsl"],
    summary="Nhận dạng speech và ghép video VSL",
)
async def translate_speech_to_vsl_video(
    file: UploadFile = File(description="File ghi âm tiếng Việt"),
) -> SpeechToVSLVideoResponse:
    suffix, content = await read_upload(file, AUDIO_SUFFIXES, 25, "speech.webm")
    try:
        with tempfile.TemporaryDirectory(prefix="signmate-speech-") as directory:
            audio_path = Path(directory) / f"speech{suffix}"
            audio_path.write_bytes(content)
            transcript = await run_in_threadpool(speech_recognizer.transcribe, audio_path)
        result = await run_in_threadpool(text_to_vsl_video_service.translate, transcript)
        return SpeechToVSLVideoResponse(
            transcript=transcript,
            speech_model=speech_recognizer.model_name,
            **result,
        )
    except AudioDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeechRecognizerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaCloudError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DatasetUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VideoAssemblyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/api/v1/translate/video-to-text",
    response_model=VideoToTextResponse,
    tags=["video-to-text"],
    summary="Nhận diện video VSL và tạo câu tiếng Việt",
)
async def translate_video_to_text(
    file: UploadFile = File(description="Video chứa một hoặc nhiều ký hiệu VSL"),
    segment_seconds: float = Form(default=4.0, ge=1.0, le=15.0),
    expected_signs: int = Form(default=0, ge=0, le=30),
    min_confidence: float = Form(default=0.35, ge=0, le=1),
) -> VideoToTextResponse:
    suffix, content = await read_upload(file, VIDEO_SUFFIXES, 100, "video.mp4")
    try:
        with tempfile.TemporaryDirectory(prefix="signmate-video-text-") as directory:
            video_path = Path(directory) / f"video{suffix}"
            video_path.write_bytes(content)
            result = await run_in_threadpool(
                video_to_text_service.translate,
                video_path,
                segment_seconds,
                expected_signs,
                min_confidence,
            )
        return VideoToTextResponse(filename=file.filename or "video", **result)
    except VideoDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NoSignsRecognized as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RecognizerUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Model chưa sẵn sàng: {exc}") from exc
    except OllamaCloudError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
