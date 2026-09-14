from __future__ import annotations

import csv
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = PROJECT_ROOT / "data" / "kaggle_metadata" / "labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kaggle_dataset" / "videos"
DATASET = "aresusayhi/vsl-vietnamese-sign-languages"
QIPEDC_VIDEO_BASE = "https://qipedc.moet.gov.vn/videos"


def download_from_qipedc(filename: str) -> tuple[bool, str]:
    target = OUTPUT_DIR / filename
    partial = target.with_suffix(target.suffix + ".part")
    process = subprocess.run(
        [
            "curl", "-sSL", "--fail", "--connect-timeout", "30",
            "--max-time", "300", "--retry", "2", "--retry-delay", "3",
            "-o", str(partial), f"{QIPEDC_VIDEO_BASE}/{filename}",
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode == 0 and partial.is_file() and partial.stat().st_size > 0:
        partial.replace(target)
        return True, "downloaded-from-qipedc"
    partial.unlink(missing_ok=True)
    return False, process.stderr[-500:]


def download(filename: str) -> tuple[str, bool, str]:
    target = OUTPUT_DIR / filename
    if target.is_file() and target.stat().st_size > 0:
        return filename, True, "cached"
    source = os.getenv("VSL_DOWNLOAD_SOURCE", "qipedc").lower()
    command = ["kaggle", "datasets", "download", DATASET, "-f",
               f"Dataset/Videos/{filename}", "-p", str(OUTPUT_DIR), "--unzip", "-q"]
    max_attempts = int(os.getenv("KAGGLE_DOWNLOAD_RETRIES", "8"))
    for attempt in range(1, max_attempts + 1):
        if source == "qipedc":
            ok, detail = download_from_qipedc(filename)
        else:
            process = subprocess.run(command, capture_output=True, text=True)
            ok = process.returncode == 0 and (OUTPUT_DIR / filename).is_file()
            detail = f"downloaded-attempt-{attempt}" if ok else process.stderr[-500:]
        if ok:
            return filename, True, detail
        if attempt < max_attempts:
            # Both sources can temporarily throttle or time out after bursts.
            # A bounded exponential pause lets the same run recover by itself.
            delay = min(60, 5 * (2 ** (attempt - 1))) + random.random() * 2
            time.sleep(delay)
    return filename, False, detail


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS_FILE.open(encoding="utf-8-sig", newline="") as stream:
        filenames = list(dict.fromkeys(row["VIDEO"].strip() for row in csv.DictReader(stream)))
    filenames = [name for name in filenames if name]
    workers = int(os.getenv("KAGGLE_DOWNLOAD_WORKERS", "4"))
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, name): name for name in filenames}
        for future in as_completed(futures):
            filename, ok, detail = future.result()
            completed += 1
            if not ok:
                failures.append((filename, detail))
            if completed % 100 == 0 or completed == len(filenames):
                print(
                    f"{completed}/{len(filenames)} | lỗi: {len(failures)}",
                    flush=True,
                )
    if failures:
        failure_file = PROJECT_ROOT / "data" / "kaggle_metadata" / "download_failures.txt"
        failure_file.write_text(
            "\n".join(f"{name}\t{detail}" for name, detail in failures),
            encoding="utf-8",
        )
        raise SystemExit(f"Còn {len(failures)} file lỗi; xem {failure_file}")
    print(f"Đã tải đủ {len(filenames)} video vào {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
