from __future__ import annotations

import csv
import hashlib
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

from app.ollama_normalizer import OllamaCloudError, normalized_key, ollama_normalizer


BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_FILE = BASE_DIR / "data" / "kaggle_metadata" / "labels.csv"
FULL_DATA_DIR = BASE_DIR / "data" / "kaggle_dataset" / "videos"
GENERATED_DIR = BASE_DIR / "static" / "generated"


class DatasetUnavailable(RuntimeError):
    pass


class VideoAssemblyError(RuntimeError):
    pass


def _tokens(value: str) -> set[str]:
    return {token for token in normalized_key(value).split() if len(token) > 1}


class KaggleVSLDictionary:
    def __init__(self) -> None:
        self._labels: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._labels is not None:
            return self._labels
        labels: dict[str, dict] = {}
        with LABELS_FILE.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                label = row["LABEL"].strip()
                filename = row["VIDEO"].strip()
                key = normalized_key(label)
                entry = labels.setdefault(key, {"label": label, "videos": []})
                if filename and filename not in entry["videos"]:
                    entry["videos"].append(filename)
        self._labels = labels
        return labels

    def candidates(self, text: str, limit: int = 250) -> list[str]:
        labels = self._load()
        text_key = normalized_key(text)
        text_tokens = _tokens(text)
        scored = []
        for key, entry in labels.items():
            label_tokens = _tokens(key)
            overlap = len(text_tokens & label_tokens)
            exact_bonus = 3.0 if f" {key} " in f" {text_key} " else 0.0
            coverage = overlap / max(len(label_tokens), 1)
            fuzzy = SequenceMatcher(None, text_key, key).ratio()
            score = exact_bonus + coverage * 2 + fuzzy
            if exact_bonus or overlap or fuzzy >= 0.45:
                scored.append((score, len(key), entry["label"]))
        scored.sort(reverse=True)
        return [item[2] for item in scored[:limit]]

    def longest_matches(self, text: str) -> tuple[list[str], list[str]]:
        labels = self._load()
        words = normalized_key(text).split()
        phrases = {
            tuple(key.split()): entry["label"]
            for key, entry in labels.items()
            if key
        }
        max_length = max((len(phrase) for phrase in phrases), default=1)
        matched: list[str] = []
        unknown: list[str] = []
        index = 0
        while index < len(words):
            found = None
            found_length = 0
            for length in range(min(max_length, len(words) - index), 0, -1):
                phrase = tuple(words[index : index + length])
                if phrase in phrases:
                    found = phrases[phrase]
                    found_length = length
                    break
            if found:
                matched.append(found)
                index += found_length
            else:
                unknown.append(words[index])
                index += 1
        return matched, unknown

    def _choose_video(self, label: str) -> str:
        entry = self._load().get(normalized_key(label))
        if not entry:
            raise DatasetUnavailable(f"Không có nhãn trong dataset: {label}")
        return entry["videos"][0]

    def _materialize(self, filename: str) -> Path:
        video = FULL_DATA_DIR / filename
        if video.is_file() and video.stat().st_size > 0:
            return video
        raise DatasetUnavailable(
            f"Thiếu {filename}; hãy tải dataset vào data/kaggle_dataset/videos"
        )

    def resolve(self, labels: list[str]) -> list[dict]:
        resolved = []
        for label in labels:
            filename = self._choose_video(label)
            resolved.append({"label": label, "filename": filename, "path": self._materialize(filename)})
        return resolved


def assemble_videos(clips: list[dict], cache_key: str) -> Path:
    if not clips:
        raise VideoAssemblyError("Không có video để ghép")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output = GENERATED_DIR / f"{cache_key}.mp4"
    if output.is_file():
        return output
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise VideoAssemblyError("Thiếu imageio-ffmpeg") from exc

    command = [ffmpeg, "-y"]
    for clip in clips:
        command.extend(["-i", str(clip["path"])])
    filters = []
    streams = []
    for index in range(len(clips)):
        name = f"v{index}"
        filters.append(
            f"[{index}:v]scale=640:480:force_original_aspect_ratio=decrease,"
            f"pad=640:480:(ow-iw)/2:(oh-ih)/2:black,fps=25,"
            f"setsar=1,setpts=PTS-STARTPTS[{name}]"
        )
        streams.append(f"[{name}]")
    filters.append(f"{''.join(streams)}concat=n={len(clips)}:v=1:a=0[outv]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        output.unlink(missing_ok=True)
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")[-1000:]
        raise VideoAssemblyError(f"Không ghép được video: {detail}") from exc
    return output


class TextToVSLVideoService:
    def __init__(self) -> None:
        self.dictionary = KaggleVSLDictionary()

    def translate(self, text: str) -> dict:
        candidates = self.dictionary.candidates(text)
        normalized = ollama_normalizer.normalize(text, candidates)
        if not normalized["units"]:
            fallback_units, fallback_unknown = self.dictionary.longest_matches(text)
            normalized["units"] = fallback_units
            normalized["unknown_words"] = list(
                dict.fromkeys([*normalized["unknown_words"], *fallback_unknown])
            )
        if not normalized["units"]:
            raise OllamaCloudError("AI không tìm được nhãn video phù hợp")
        if len(normalized["units"]) > 30:
            raise VideoAssemblyError("Tối đa 30 ký hiệu trong một lần ghép")
        clips = self.dictionary.resolve(normalized["units"])
        digest = hashlib.sha256(
            "\0".join(item["filename"] for item in clips).encode("utf-8")
        ).hexdigest()[:20]
        output = assemble_videos(clips, digest)
        labels = normalized.pop("units")
        return {
            **normalized,
            "labels": labels,
            "clips": [
                {"label": item["label"], "filename": item["filename"]} for item in clips
            ],
            "video_url": f"/static/generated/{output.name}",
        }


text_to_vsl_video_service = TextToVSLVideoService()
