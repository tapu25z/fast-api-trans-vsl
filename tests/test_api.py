from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def vsl_video_result(text: str) -> dict:
    return {
        "normalized_text": text,
        "labels": [text],
        "unknown_words": [],
        "model": "gpt-oss:120b",
        "clips": [{"label": text, "filename": "W03990.mp4"}],
        "video_url": "/static/generated/test.mp4",
    }


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.4.0"}


def test_text_to_vsl_video(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.text_to_vsl_video_service.translate",
        lambda text: vsl_video_result(text),
    )
    response = client.post(
        "/api/v1/translate/text-to-vsl-video",
        json={"text": "xin lỗi"},
    )
    assert response.status_code == 200
    assert response.json()["labels"] == ["xin lỗi"]


def test_blank_text_is_rejected() -> None:
    response = client.post(
        "/api/v1/translate/text-to-vsl-video",
        json={"text": "   "},
    )
    assert response.status_code == 422


def test_speech_to_vsl_video(monkeypatch) -> None:
    monkeypatch.setattr("app.main.speech_recognizer.transcribe", lambda path: "xin lỗi")
    monkeypatch.setattr(
        "app.main.text_to_vsl_video_service.translate",
        lambda text: vsl_video_result(text),
    )
    response = client.post(
        "/api/v1/translate/speech-to-vsl-video",
        files={"file": ("speech.webm", b"fake-audio", "audio/webm")},
    )
    assert response.status_code == 200
    assert response.json()["transcript"] == "xin lỗi"
    assert response.json()["labels"] == ["xin lỗi"]


def test_speech_rejects_wrong_extension() -> None:
    response = client.post(
        "/api/v1/translate/speech-to-vsl-video",
        files={"file": ("speech.txt", b"not-audio", "text/plain")},
    )
    assert response.status_code == 415


def test_video_to_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.video_to_text_service.translate",
        lambda path, segment_seconds, expected_signs, min_confidence: {
            "text": "Xin lỗi cảm ơn",
            "labels": ["Xin lỗi", "Cảm ơn"],
            "segments": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "label": "Xin lỗi",
                    "confidence": 0.91,
                    "accepted": True,
                    "top_predictions": [{"label": "Xin lỗi", "confidence": 0.91}],
                }
            ],
            "duration_seconds": 3.0,
            "segment_seconds": 3.0,
            "recognition_model": "test-videomae",
            "language_model": "gpt-oss:120b",
        },
    )
    response = client.post(
        "/api/v1/translate/video-to-text",
        files={"file": ("sentence.mp4", b"fake-video", "video/mp4")},
        data={"expected_signs": "2"},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "Xin lỗi cảm ơn"
    assert response.json()["labels"] == ["Xin lỗi", "Cảm ơn"]


def test_video_rejects_wrong_extension() -> None:
    response = client.post(
        "/api/v1/translate/video-to-text",
        files={"file": ("video.txt", b"not-video", "text/plain")},
    )
    assert response.status_code == 415


def test_dictionary_longest_match() -> None:
    from app.text_to_vsl_video import KaggleVSLDictionary

    matched, unknown = KaggleVSLDictionary().longest_matches("Tôi đi học")
    assert matched == ["đi học"]
    assert unknown == ["tôi"]
