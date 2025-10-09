import google.generativeai as genai  # thư viện Google Gemini để gọi model sinh nội dung (LLM)
import json, re, logging, time        # thư viện chuẩn: JSON, regex, logging, và đo thời gian
from io import StringIO               # dùng để tạo buffer tạm lưu text khi trích xuất từ PDF
from pdfminer.high_level import extract_text_to_fp  # hàm trích xuất nội dung PDF thành text
from pdfminer.layout import LAParams  # tham số điều chỉnh cách parse layout PDF
from qdrant_client import QdrantClient              # client kết nối Qdrant vector database
from qdrant_client.models import VectorParams, Distance  # định nghĩa cấu hình vector cho Qdrant
from sentence_transformers import SentenceTransformer    # model tạo embedding từ văn bản

logger = logging.getLogger(__name__)  # tạo logger theo tên module, phục vụ ghi log lỗi hoặc thông tin


# hàm chuyển PDF sang HTML text
def pdf_to_text(pdf_path: str) -> str:

    start_time = time.time()

    output = StringIO()
    laparams = LAParams(line_margin=0.5, word_margin=0.1)  # điều chỉnh cách nhận diện dòng và từ
    with open(pdf_path, "rb") as f:
        extract_text_to_fp(f, output, laparams=laparams, output_type="html", codec=None)

    print(f"[pdf_to_text]: {time.time() - start_time:.2f}s")

    return output.getvalue()


# hàm trích xuất thông tin paper bằng Gemini
def extract_paper_info(html_content: str, gemini_model) -> dict:

    """
    Dùng Gemini model để trích xuất thông tin học thuật (title, authors, abstract, ...)
    từ nội dung HTML đã parse từ PDF
    """

    start_time = time.time()

    # prompt yêu cầu Gemini trả về JSON chứa đầy đủ thông tin của bài báo
    prompt = f"""
    Extract from this academic paper HTML. Return ONLY valid JSON:
    {{
        "title": "",
        "authors": [],
        "email": "", 
        "doi": "", 
        "journal": "", 
        "year": "", 
        "abstract": "", 
        "keywords": [], 
        "introduction": "", 
        "methodology": "",
        "results": "", 
        "discussion": "", 
        "conclusion": "", 
        "references": []
    }}
        HTML:
        {html_content}
        JSON only:
    """
    try:
        # gọi model Gemini để sinh nội dung theo prompt
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)  # temperature thấp giúp kết quả ổn định, ít "bịa"
        )

        # làm sạch text trả về, loại bỏ ```json``` nếu có
        text = re.sub(r"^```(?:json)?\s*|```$", "", (response.text or "").strip(), 
                      flags=re.MULTILINE
        )

        # tìm đoạn JSON thực sự trong text
        match = re.search(r"\{.*\}", text, re.DOTALL)

        print(f"[extract_paper_info]: {time.time() - start_time:.2f}s")

        # parse JSON thành Python dict, nếu không có thì trả về rỗng
        return json.loads(match.group()) if match else {}

    except Exception as e:
        logger.error(f"Gemini extract error: {e}")
        return {}


# hàm chunk hóa nội dung paper cho RAG
def chunk_text_rag(data: dict, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:

    """
    Chia nhỏ (chunk) các phần văn bản của bài báo để lưu vào vector DB (RAG)
    Mỗi chunk gồm nội dung + metadata (title, authors, section, ...)
    """

    start_time = time.time()

    chunks = []

    # lấy thông tin metadata cơ bản
    base = {
        "title": data.get("title", ""),
        "authors": ", ".join(data.get("authors", [])),
        "email": data.get("email", ""),
        "doi": data.get("doi", ""),
        "journal": data.get("journal", ""),
        "year": data.get("year", ""),
        "keywords": data.get("keywords", []),
    }

    # tạo text metadata riêng cho embedding (chunk đầu chứa thông tin cơ bản)
    meta_text = f"""Title: {base['title']}
                    Authors: {base['authors']}
                    Email: {base['email']}
                    DOI: {base['doi']}
                    Journal: {base['journal']}
                    Year: {base['year']}
                    Keywords: {', '.join(base['keywords'])}"""
    
    chunks.append({"text_for_embedding": meta_text,
                    "metadata": {**base, "section": "metadata"}}
    )

    # duyệt qua từng phần chính của paper và chunk hóa theo kích thước cố định
    for section_name in ["abstract", 
                         "introduction", 
                         "methodology", 
                         "results", 
                         "discussion", 
                         "conclusion"]:
        
        text = data.get(section_name, "")

        if not text:
            continue

        words = text.split()
        step = chunk_size - chunk_overlap  # đảm bảo các chunk có vùng chồng lấn để tránh mất ngữ cảnh        
        
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text_for_embedding": chunk_text.strip(),
                "metadata": {**base, "section": section_name}
            })

    # xử lý riêng phần tài liệu tham khảo
    refs = data.get("references", [])

    if refs:
        refs_text = "\n".join(refs) if isinstance(refs, list) else str(refs)
        words = refs_text.split()
        step = chunk_size - chunk_overlap

        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text_for_embedding": chunk_text.strip(),
                "metadata": {**base, "section": "references"}
            })

    print(f"[chunk_text_rag]: {time.time() - start_time:.2f}s")

    return chunks


# hàm tạo collection Qdrant mới
def setup_qdrant(collection: str = "papers") -> tuple:

    """
    Khởi tạo kết nối Qdrant và model embedding
    Nếu collection đã tồn tại, nó sẽ bị ghi đè (recreate)
    """

    start_time = time.time()

    qdrant = QdrantClient(host="localhost", port=6333)
    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

    # tạo collection với vector 1024 chiều, dùng cosine distance để đo độ tương tự
    qdrant.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    print(f"[setup_qdrant]: {time.time() - start_time:.2f}s")

    return qdrant, embedder


# hàm upload các chunk vào Qdrant
def upload_chunks(qdrant, embedder, chunks: list, collection: str = "papers"):

    """Sinh vector embedding và upload từng chunk cùng metadata vào Qdrant"""

    start_time = time.time()

    texts = [c["text_for_embedding"] for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=True)  # sinh vector embeddings
    payloads = [{**c["metadata"], 
                 "text": c["text_for_embedding"]} 
                 for c in chunks
    ]

    qdrant.upload_collection(
        collection_name=collection,
        vectors=vectors,
        payload=payloads,
        ids=list(range(len(chunks))),
        batch_size=64
    )

    print(f"[upload_chunks]: Đã upload {len(chunks)} chunks trong {time.time() - start_time:.2f}s")


# tìm các đoạn tương tự (retrieval)
def search_similar_chunks(qdrant, embedder, query: str, collection: str = "papers", top_k: int = 5) -> list:

    """Tìm các đoạn văn bản trong Qdrant có embedding tương tự nhất với câu hỏi"""

    query_vec = embedder.encode(query)
    results = qdrant.search(collection_name=collection, 
                            query_vector=query_vec, 
                            limit=top_k, 
                            with_payload=True
    )
    
    return [
        {"score": round(hit.score, 4), 
         "section": hit.payload.get("section", ""), 
         "text": hit.payload.get("text", "")}
        for hit in results
    ]


# hàm sinh câu trả lời bằng Gemini
def generate_answer(gemini_model, query: str, chunks: list) -> str:

    """Dùng Gemini model để sinh câu trả lời dựa trên các đoạn liên quan"""

    if not chunks:
        return "Không tìm thấy thông tin liên quan"
    
    context = "\n\n".join([f"[{c['section']}] {c['text']}" for c in chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = gemini_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))

    return response.text.strip()


# hàm pipeline truy vấn hoàn chỉnh
def query_paper(qdrant, embedder, gemini_model, question: str, collection: str = "papers"):

    """Pipeline truy vấn: tìm → tổng hợp → sinh câu trả lời"""

    chunks = search_similar_chunks(qdrant, embedder, question, collection)
    answer = generate_answer(gemini_model, question, chunks)

    return answer, chunks


# main
if __name__ == "__main__":

    API_KEY = ""
    PDF_FILE = "test_pdf_3.pdf"
    pdf_name = PDF_FILE.replace(".pdf", "")

    qdrant = QdrantClient(host="localhost", port=6333)
    collections = [c.name for c in qdrant.get_collections().collections]

    # nếu collection chưa có, thực thi toàn bộ pipeline build
    if pdf_name not in collections:
        print(f"\n Tạo collection mới: {pdf_name}")
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")

        html = pdf_to_text(PDF_FILE)
        paper_data = extract_paper_info(html, gemini_model)
        chunks = chunk_text_rag(paper_data)

        qdrant, embedder = setup_qdrant(pdf_name)
        upload_chunks(qdrant, embedder, chunks, pdf_name)

        print(f"\n Collection '{pdf_name}' tạo thành công")

    # nếu đã tồn tại collection, chỉ cần load model để truy vấn
    else:
        print(f"\n Collection '{pdf_name}' đã tồn tại. Đang load model...")
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

    # giao diện hỏi đáp qua terminal
    while True:

        question = input("\n Nhập câu hỏi (hoặc 'exit' để thoát): ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Thoát chương trình.")
            break

        try:
            answer, related_chunks = query_paper(qdrant, embedder, gemini_model, question, pdf_name)
            print("\n Trả lời:\n", answer)
            print("\n Các đoạn liên quan:")
            for i, c in enumerate(related_chunks, 1):
                print(f"{i}. ({c['section']}) {c['text'][:250]}...\n")

        except Exception as e:
            print(f"Lỗi khi truy vấn: {e}")
