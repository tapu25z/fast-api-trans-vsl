from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import httpx
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)
DEFAULT_MODEL = "gpt-oss:120b"


class OllamaCloudError(RuntimeError):
    pass


def normalized_key(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold().strip()
    value = re.sub(r"[^\w\sđ]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value)


def get_api_key() -> str:
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    if key:
        return key
    key_file = BASE_DIR / "api.txt"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise OllamaCloudError("Thiếu OLLAMA_API_KEY: điền key vào .env, biến môi trường hoặc file api.txt")
    return key


class OllamaSentenceNormalizer:
    def __init__(self) -> None:
        self.host = os.getenv("OLLAMA_HOST", "https://ollama.com").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    def normalize(self, text: str, available_labels: list[str]) -> dict:
        if not available_labels:
            raise OllamaCloudError("Không tìm được nhãn ứng viên trong dataset")

        prompt = {
            "input_text": text,
            "available_labels": available_labels,
            "rules": [
                "Tách câu thành các đơn vị nghĩa để ghép video ngôn ngữ ký hiệu Việt Nam.",
                "Mỗi phần tử units phải giống chính xác một chuỗi trong available_labels.",
                "Ưu tiên cụm từ dài nhất và giữ nguyên ý nghĩa câu.",
                "Không tự tạo nhãn mới. Từ không biểu diễn được đưa vào unknown_words.",
                "Giữ thứ tự dễ hiểu; không tuyên bố đây là chuyển đổi ngữ pháp VSL hoàn chỉnh.",
            ],
            "response_schema": {
                "normalized_text": "string",
                "units": ["exact available label"],
                "unknown_words": ["string"],
            },
        }
        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                headers={"Authorization": f"Bearer {get_api_key()}"},
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Bạn là bộ chuẩn hóa câu tiếng Việt có ràng buộc từ điển. "
                                "Chỉ trả về một JSON object, không markdown và không giải thích."
                            ),
                        },
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                    "options": {"temperature": 0},
                },
                timeout=90,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            result = json.loads(content)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise OllamaCloudError(f"Ollama Cloud trả về lỗi: {exc}") from exc

        allowed = {normalized_key(label): label for label in available_labels}
        units = []
        rejected = []
        for unit in result.get("units", []):
            key = normalized_key(str(unit))
            if key in allowed:
                units.append(allowed[key])
            else:
                rejected.append(str(unit))
        unknown = [str(item) for item in result.get("unknown_words", [])]
        unknown.extend(rejected)
        return {
            "normalized_text": str(result.get("normalized_text") or text),
            "units": units,
            "unknown_words": list(dict.fromkeys(unknown)),
            "model": self.model,
        }

    def labels_to_text(self, labels: list[str]) -> str:
        if not labels:
            raise OllamaCloudError("Không có nhãn VSL để tạo câu")
        prompt = {
            "recognized_vsl_labels": labels,
            "rules": [
                "Chuyển chuỗi nhãn nhận diện từ video VSL thành một câu tiếng Việt tự nhiên.",
                "Giữ đúng nội dung và thứ tự chính của các nhãn.",
                "Không tự thêm thông tin không xuất hiện trong nhãn.",
                "Chỉ trả về JSON theo schema.",
            ],
            "response_schema": {"text": "string"},
        }
        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                headers={"Authorization": f"Bearer {get_api_key()}"},
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Bạn là bộ hậu xử lý kết quả nhận diện ngôn ngữ ký hiệu Việt Nam. "
                                "Chỉ trả về một JSON object, không markdown và không giải thích."
                            ),
                        },
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                    "options": {"temperature": 0},
                },
                timeout=90,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            result = json.loads(content)
            text = str(result.get("text") or "").strip()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise OllamaCloudError(f"Ollama Cloud trả về lỗi: {exc}") from exc
        if not text:
            return " ".join(labels)
        return text


ollama_normalizer = OllamaSentenceNormalizer()
