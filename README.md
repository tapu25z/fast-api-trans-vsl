# FastAPI Vietnamese Sign Language (VSL)

API thử nghiệm chuyển đổi giữa tiếng Việt và Ngôn ngữ ký hiệu Việt Nam.

Project có **2 luồng chính**:

1. **Speech hoặc Text → VSL**: nhận câu tiếng Việt, trả danh sách nhãn VSL và
   ghép các video ký hiệu thành một video để hiển thị.
2. **Video VSL → Text**: tách video thành các đoạn, nhận diện từng ký hiệu và
   ghép các nhãn thành câu tiếng Việt.

> Đây là MVP nghiên cứu. Model nhận diện video hiện chỉ biết 100 nhãn, không
> nhận diện được toàn bộ 3.314 nhãn của dataset Kaggle nếu chưa huấn luyện lại.

## Các model đang dùng

| Thành phần | Model | Chức năng |
|---|---|---|
| Nhận dạng giọng nói | `openai/whisper-small` | Speech tiếng Việt → text |
| Xử lý ngôn ngữ | Ollama Cloud `gpt-oss:120b` | Chuẩn hóa câu, chọn/ghép nhãn thành câu |
| Nhận diện video VSL | `star092304/vi-sign-language-videomae-base` | Video một ký hiệu → một trong 100 nhãn |

Có thể hiểu project gồm hai model nghiệp vụ:

```text
Model/luồng 1: Speech hoặc Text → VSL labels → ghép video
Model/luồng 2: Video VSL → labels → câu tiếng Việt
```

## 1. Cài đặt

Yêu cầu Python 3.11 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ml.txt
```

## 2. Cấu hình Ollama Cloud

Tạo biến môi trường `OLLAMA_API_KEY` trước khi chạy API.

macOS/Linux:

```bash
export OLLAMA_API_KEY="YOUR_OLLAMA_CLOUD_KEY"
```

Windows PowerShell:

```powershell
$env:OLLAMA_API_KEY="YOUR_OLLAMA_CLOUD_KEY"
```

Model mặc định là `gpt-oss:120b`. Có thể đổi bằng biến `OLLAMA_MODEL`. Không
commit hoặc gửi API key lên Git.

## 3. Tải dataset VSL từ Kaggle

Dataset: [VSL - Vietnamese Sign Languages trên Kaggle](https://www.kaggle.com/datasets/aresusayhi/vsl-vietnamese-sign-languages)

Dataset có 4.362 video và 3.314 nhãn sau chuẩn hóa. Project cần đúng cấu trúc:

```text
data/kaggle_metadata/labels.csv
data/kaggle_dataset/videos/*.mp4
```

### Cách A — dùng script của project

File `labels.csv` đã có sẵn trong project. Chạy:

```bash
python scripts/download_kaggle_dataset.py
```

Script tải đúng 4.362 video, tự bỏ qua file đã có và tiếp tục được nếu bị ngắt.

### Cách B — tải trực tiếp bằng Kaggle CLI

Cài và đăng nhập Kaggle CLI, sau đó chạy:

```bash
kaggle datasets download \
  -d aresusayhi/vsl-vietnamese-sign-languages \
  -p data/kaggle_download \
  --unzip
```

Sao chép dữ liệu về đúng cấu trúc project:

```bash
mkdir -p data/kaggle_dataset/videos data/kaggle_metadata
cp data/kaggle_download/Dataset/Videos/*.mp4 data/kaggle_dataset/videos/
cp data/kaggle_download/Dataset/Labels/label.csv data/kaggle_metadata/labels.csv
```

Nếu tên `label.csv` hoặc vị trí sau giải nén khác, tìm file chứa các cột
`VIDEO` và `LABEL`, rồi đổi tên thành `data/kaggle_metadata/labels.csv`.

Kiểm tra số video:

```bash
find data/kaggle_dataset/videos -name '*.mp4' | wc -l
```

Kết quả mong đợi: `4362`.

## 4. Chạy FastAPI

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở:

- Demo: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

Lần chạy đầu, Whisper và VideoMAE sẽ được tải từ Hugging Face. Các lần sau dùng
model đã cache trên máy.

## 5. Speech hoặc Text → VSL

### Pipeline

```text
Speech
  → Whisper chuyển thành text
  → gpt-oss:120b chuẩn hóa/tách câu
  → đối chiếu labels.csv
  → danh sách labels
  → lấy các MP4 tương ứng từ dataset
  → FFmpeg ghép video
  → trả labels + video_url để giao diện phát video

Text
  → gpt-oss:120b chuẩn hóa/tách câu
  → các bước còn lại giống Speech
```

### Text → VSL

```http
POST /api/v1/translate/text-to-vsl-video
Content-Type: application/json
```

Request:

```json
{"text": "địa chỉ tỉnh"}
```

Response mẫu:

```json
{
  "normalized_text": "địa chỉ tỉnh",
  "labels": ["địa chỉ", "tỉnh"],
  "unknown_words": [],
  "model": "gpt-oss:120b",
  "clips": [
    {"label": "địa chỉ", "filename": "D0001N.mp4"},
    {"label": "tỉnh", "filename": "D0002.mp4"}
  ],
  "video_url": "/static/generated/cfd97c3b08649751f140.mp4"
}
```

Backend chính chỉ cần đọc trường `labels`. `video_url` là đường dẫn tương đối để
frontend cùng server phát video đã ghép.

### Speech → VSL

```http
POST /api/v1/translate/speech-to-vsl-video
Content-Type: multipart/form-data
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/translate/speech-to-vsl-video \
  -F 'file=@ghi-am.wav'
```

Hỗ trợ WAV, MP3, M4A, AAC, OGG, WebM và MP4. Response giống Text → VSL và có
thêm `transcript` cùng `speech_model`.

## 6. Video VSL → Text

### Pipeline

```text
Video VSL
  → kiểm tra định dạng, dung lượng, thời lượng
  → chia video thành các đoạn ký hiệu
  → lấy 16 frame cho mỗi đoạn
  → VideoMAE dự đoán top-3 nhãn
  → bỏ kết quả dưới min_confidence
  → gộp các nhãn trùng liền nhau
  → gpt-oss:120b chuyển chuỗi nhãn thành câu tiếng Việt
  → trả text + labels + kết quả từng segment
```

Endpoint:

```http
POST /api/v1/translate/video-to-text
Content-Type: multipart/form-data
```

Ví dụ:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/translate/video-to-text \
  -F 'file=@video-cau-vsl.mp4' \
  -F 'expected_signs=2' \
  -F 'segment_seconds=4' \
  -F 'min_confidence=0.35'
```

Tham số:

- `expected_signs`: số ký hiệu trong video; `0` để hệ thống tự ước lượng.
- `segment_seconds`: độ dài dự kiến của một ký hiệu, mặc định 4 giây.
- `min_confidence`: ngưỡng giữ lại dự đoán, mặc định `0.35`.

Response mẫu:

```json
{
  "filename": "vsl-two-signs.mp4",
  "text": "Xin lỗi, ban ngày.",
  "labels": ["Xin lỗi", "Ban ngày"],
  "segments": [
    {
      "index": 0,
      "start_seconds": 0.0,
      "end_seconds": 3.98,
      "label": "Xin lỗi",
      "confidence": 0.748855,
      "accepted": true,
      "top_predictions": []
    }
  ],
  "duration_seconds": 7.96,
  "segment_seconds": 3.98,
  "recognition_model": "star092304/vi-sign-language-videomae-base",
  "language_model": "gpt-oss:120b"
}
```

## 7. Giao diện demo

Trang `http://127.0.0.1:8000` có hai khu vực:

- **Speech hoặc Text → video VSL**: nhập text, ghi âm hoặc chọn file âm thanh;
  frontend đọc `video_url` và phát video đã ghép.
- **Video VSL → Text**: chọn video, API trả câu và danh sách nhãn để in lên màn
  hình.

## 8. Kiểm thử

```bash
python -m pytest -q
```

## Giới hạn cần biết

- VideoMAE hiện chỉ có 100 nhãn. Ví dụ `D0002.mp4` có nhãn thật là `tỉnh`, nhưng
  model 100 lớp không có nhãn `tỉnh` nên có thể dự đoán sai thành `Cứu`.
- Có đủ 4.362 video giúp luồng Text/Speech → VSL ghép được nhiều từ hơn, nhưng
  không tự làm VideoMAE biết 3.314 nhãn.
- Muốn Video → Text nhận diện toàn bộ dataset cần huấn luyện lại model phân loại
  với toàn bộ nhãn, hoặc xây dựng mô hình nhận diện chuỗi VSL chuyên dụng.
- Bản MVP chia video theo thời gian. Nếu biết số ký hiệu, nên truyền
  `expected_signs` để có kết quả ổn định hơn.
