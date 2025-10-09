# ReadPaperApp

Ứng dụng nhỏ dạng command-line để trích xuất thông tin từ một bài báo (PDF), đưa vào cơ sở dữ liệu vector Qdrant và trả lời câu hỏi bằng cách kết hợp Retrieval + Generation (RAG) với Google Gemini.

Tập tin chính: `app.py`.

## Tổng quan hoạt động

- Chuyển PDF sang HTML text bằng `pdfminer.six`.
- Gọi model Gemini (Google Generative AI) để trích xuất thông tin có cấu trúc của bài báo (title, authors, abstract, introduction, methodology, results, discussion, conclusion, references).
- Chia nhỏ (chunk) các phần văn bản, tạo embedding bằng `sentence-transformers` và lưu vector + metadata vào Qdrant.
- Từ giao diện dòng lệnh, người dùng nhập câu hỏi → ứng dụng tìm các đoạn tương tự trong Qdrant → gửi context cho Gemini để sinh câu trả lời.

## Yêu cầu

- Python 3.10+ (khuyến nghị)
- Qdrant (chạy local hoặc remote). Mặc định `app.py` kết nối tới `localhost:6333`.
- Quyền truy cập API Google Generative AI (Gemini) và API key.
- Các thư viện Python: `pdfminer.six`, `qdrant-client`, `sentence-transformers`, `torch`, và client SDK của Google Generative AI.

Lưu ý: `sentence-transformers` phụ thuộc vào `torch` — nếu bạn muốn chạy trên GPU, cài bản `torch` phù hợp với CUDA trước.

## Cài đặt nhanh (PowerShell trên Windows)

1. Tạo và kích hoạt virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Cài đặt các thư viện cần thiết (ví dụ cơ bản):

```powershell
pip install pdfminer.six qdrant-client sentence-transformers torch transformers
# Thêm package SDK của Google Generative AI nếu tên pip phù hợp (ví dụ: google-generative-ai)
```

3. Chạy Qdrant (Docker):

```powershell
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage -d qdrant/qdrant
```

## Cấu hình (trong `app.py`)

- `API_KEY` — điền API key Gemini (hiện tại script để chuỗi rỗng `""`).
- `PDF_FILE` — tên file PDF bạn muốn index (mặc định `test_pdf_3.pdf`).
- Qdrant: mặc định `QdrantClient(host="localhost", port=6333)` — thay đổi nếu Qdrant của bạn chạy ở host/port khác.
- Gemini model: `gemini-2.5-flash` (được khởi tạo trong mã). Thay đổi nếu bạn sử dụng model khác.
- Embedding model: `Qwen/Qwen3-Embedding-0.6B` (nạp bằng `SentenceTransformer`). Bạn có thể đổi sang model embedding khác nếu cần, nhưng nhớ cập nhật kích thước vector (`VectorParams(size=1024)`).

## Cách chạy

1. Đặt file PDF vào cùng thư mục với `app.py` hoặc chỉnh đường dẫn `PDF_FILE`.
2. Mở PowerShell, kích hoạt venv và chạy:

```powershell
python app.py
```

Lần chạy đầu tiên (nếu chưa có collection Qdrant cho file PDF) sẽ:

- Chuyển PDF → HTML
- Gọi Gemini để trích xuất JSON mô tả paper
- Chunk và tạo embedding
- Tạo collection Qdrant (tên = tên file PDF không có `.pdf`) và upload vectors

Sau khi index xong, chương trình sẽ vào chế độ hỏi đáp. Gõ câu hỏi vào prompt. Gõ `exit` hoặc `quit` để thoát.

## Các điểm quan trọng trong mã

- Hàm `pdf_to_text(pdf_path)` dùng `pdfminer.six` để xuất HTML từ PDF.
- Hàm `extract_paper_info(html_content, gemini_model)` gửi prompt cho Gemini và kỳ vọng nhận được một JSON chứa các trường như `title`, `authors`, `abstract`, `introduction`, `methodology`, `results`, `discussion`, `conclusion`, `references`.
- Hàm `chunk_text_rag(data, chunk_size=1000, chunk_overlap=200)` chia văn bản thành các chunk có overlap để lưu vào vector DB.
- `setup_qdrant(collection)` tạo (recreate) collection mới với vector size = 1024 và distance = COSINE.
- `upload_chunks(...)` gọi `embedder.encode()` để sinh vector và upload vào Qdrant.
- `search_similar_chunks(...)` tìm các đoạn tương tự dựa trên embedding của câu hỏi.
- `generate_answer(...)` gửi context (các đoạn tương tự) cùng câu hỏi tới Gemini để sinh câu trả lời.

## Lưu ý & hạn chế

- Script này là một POC/demo — chưa có bảo mật cho API keys, chưa xử lý lỗi chuyên sâu, và không tối ưu cho các PDF rất lớn.
- Model LLM có thể trả về text không hoàn toàn chuẩn JSON; mã cố gắng tách JSON ra bằng regex nhưng vẫn cần kiểm tra thủ công kết quả.
- Kích thước embedding trong Qdrant phải phù hợp với model embedding bạn dùng. Nếu đổi model embedding, cập nhật `VectorParams(size=...)` tương ứng.

## Gợi ý cải tiến

- Đọc cấu hình từ biến môi trường hoặc file `.env` (API key, Qdrant host/port, tên file).
- Thêm CLI (argparse/typer) để truyền đường dẫn PDF, tên collection, và các tùy chọn model.
- Thêm logging chi tiết và unit-tests cho các hàm chunk / search.

---

Nếu bạn muốn, tôi có thể tiếp tục và:
- Tạo `requirements.txt` tự động từ các import trong `app.py`.
- Cập nhật `app.py` để đọc `API_KEY` và `QDRANT_HOST/PORT` từ biến môi trường.
