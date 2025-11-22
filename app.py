import pandas as pd
import google.generativeai as genai
import json
import re
import time
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from Dolphin.demo_page import DOLPHIN, process_document
from dotenv import load_dotenv

load_dotenv()

# config
geminiApi = os.getenv("geminiApiKey")
geminiModelName = "gemini-2.5-flash"

qdrantHost = "localhost"
qdrantPort = 6333

embeddingModel = "Qwen/Qwen3-Embedding-0.6B"
embeddingDim = 1024

dolphinModelPath = "./Dolphin/hf_model"
outputPath = "./output"

qdrantClient = None
embedder = None
geminiModel = None
dolphinModel = None

# load models
def get_qdrant():
    global qdrantClient
    if qdrantClient is None:
        print("loading qdrant client...")
        qdrantClient = QdrantClient(host=qdrantHost, port=qdrantPort, timeout=30)
        print("qdrant client loaded.")
    return qdrantClient

def get_embedder():
    global embedder
    if embedder is None:
        print(f"loading embedding model {embeddingModel}...")
        embedder = SentenceTransformer(embeddingModel, trust_remote_code=True)
        print("embedding model loaded.")
    return embedder

def get_gemini():
    global geminiModel
    if geminiModel is None:
        print("loading gemini...")
        genai.configure(api_key=geminiApi)
        geminiModel = genai.GenerativeModel(geminiModelName)
        print("gemini loaded.")
    return geminiModel

def get_dolphin():
    global dolphinModel
    if dolphinModel is None:
        print("loading dolphin model...")
        dolphinModel = DOLPHIN(dolphinModelPath)
        print("dolphin model loaded.")
    return dolphinModel

# utils
"""parse json từ response của llm"""
def parse_json(text):
    # bỏ markdown code block
    text = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.MULTILINE)
    # thử parse trực tiếp
    try:
        return json.loads(text)
    except:
        pass
    # thử tìm {} trong text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    print("error: cannot parse json")
    return None

"""retry support"""
def retry(func, max_tries=3, delay=3):
    for i in range(max_tries):
        try:
            return func()
        except Exception as e:
            print(f"error {i+1}: {e}")
            if i < max_tries - 1:
                time.sleep(delay * (i + 1))
            else:
                raise e

# collect
"""check collection exists"""
def check_collection_exists(name):
    try:
        client = get_qdrant()
        collections = client.get_collections().collections
        for c in collections:
            if c.name == name:
                return True
        return False
    except Exception as e:
        print(f"error: {e}")
        return False

"""get collection count"""
def get_collection_count(name):
    if not check_collection_exists(name):
        return 0
    try:
        client = get_qdrant()
        info = client.get_collection(name)
        return info.points_count
    except:
        return 0

"""create collection"""
def create_collection(name, recreate=False):
    client = get_qdrant()
    if check_collection_exists(name):
        if recreate:
            print(f"delete collection exists: {name}")
            client.delete_collection(name)
        else:
            print(f"collection {name} existed")
            return False
    
    print(f"tạo collection: {name}")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=embeddingDim, distance=Distance.COSINE)
    )
    return True

# pdf processing
"""get markdown path"""
def get_md_path(pdf_path):
    md_dir = Path(outputPath) / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    return md_dir / f"{Path(pdf_path).stem}.md"

"""convert pdf to markdown"""
def convert_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"not found pdf: {pdf_path}")
    md_path = get_md_path(str(pdf_path))
    # converted, skip
    if md_path.exists():
        print(f"has markdown: {md_path}")
        return md_path
    print(f"converting pdf {pdf_path.name}...")
    start = time.time()
    dolphin = get_dolphin()
    process_document(str(pdf_path), dolphin, outputPath)
    print(f"convert done: {time.time()-start:.1f}s")
    if not md_path.exists():
        raise Exception(f"failed create markdown: {md_path}")
    return md_path

"""read markdown content"""
def read_markdown(md_path):
    # waiting file exists
    timeout = 600
    waited = 0
    while not md_path.exists() and waited < timeout:
        time.sleep(2)
        waited += 2
    if not md_path.exists():
        raise TimeoutError(f"timeout: {md_path}")
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()

# extract
extract_prompt = '''You are an academic paper parser. Extract the following from the markdown below and return ONLY valid JSON:
{
    "title": str,
    "authors": list[str],
    "email": list[str],
    "doi": str,
    "journal": str,
    "year": str,
    "abstract": str,
    "keywords": list[str]
}

For each main section (## heading), add it as a key:
- If section has sub-sections (###), create nested dict
- Otherwise, include section content as string value

Markdown:
-----
%s
-----
'''

"""extract info from markdown"""
def extract_info(md_content):
    print("extracting paper info...")
    start = time.time()
    full_prompt = extract_prompt % md_content
    gemini = get_gemini()
    def call_api():
        resp = gemini.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        if not resp or not resp.text:
            raise Exception("gemini response empty")
        return resp.text
    response_text = retry(call_api, max_tries=3)
    data = parse_json(response_text)
    if not data:
        raise Exception("failed to parse extracted result")
    
    print(f"extracted: {time.time()-start:.1f}s")
    return data

# chunking
fields_main = ["title", "authors", "email", "doi", "journal", "year"]
"""create chunks from paper data"""
def create_chunks(data):
    print("creating chunks...")
    chunks = []
    # lấy metadata
    meta = {}
    for k in fields_main:
        meta[k] = data.get(k, "")
    # chunk 1: metadata
    meta_text = ""
    for k in fields_main:
        meta_text += f"{k.title()}: {meta[k]}\n"
    chunks.append({
        "text": meta_text,
        "metadata": {"title": meta["title"], "doi": meta["doi"], "section": "metadata"}
    })
    # chunk 2: abstract + keywords
    abstract = data.get("abstract", "")
    keywords = data.get("keywords", [])
    if isinstance(keywords, list):
        keywords = ", ".join(keywords)
    abs_text = f"abstract: {abstract}\nkeywords: {keywords}"
    chunks.append({
        "text": abs_text,
        "metadata": {"title": meta["title"], "doi": meta["doi"], "section": "abstract+keywords"}
    })
    #  other sections
    skip_keys = set(fields_main + ["abstract", "keywords"])
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, dict):
            # has sub-sections
            for sub_key, sub_value in value.items():
                chunks.append({
                    "text": f"[{key}] {sub_key}: {sub_value}",
                    "metadata": {"title": meta["title"], "doi": meta["doi"], "section": f"{key}::{sub_key}"}
                })
        else:
            chunks.append({
                "text": f"{key}: {value}",
                "metadata": {"title": meta["title"], "doi": meta["doi"], "section": key}
            })
    print(f"created: {len(chunks)} chunks")
    return chunks

# vector store
"""upload chunks to Qdrant"""
def upload_chunks(chunks, collection_name):
    print(f"uploading {len(chunks)} chunks...")
    start = time.time()
    client = get_qdrant()
    emb = get_embedder()
    # upload theo batch
    batch_size = 10
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"batch {batch_num}/{total_batches}...")
        # embed
        texts = [c["text"] for c in batch]
        vectors = emb.encode(texts, show_progress_bar=False)
        # create points
        points = []
        for j, (chunk, vec) in enumerate(zip(batch, vectors)):
            point_id = i + j
            payload = chunk["metadata"].copy()
            payload["text"] = chunk["text"]
            points.append(PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload
            ))
        # upload
        client.upsert(collection_name=collection_name, points=points)
    print(f"uploaded: {time.time()-start:.1f}s")

"""search in collection"""
def search(query, collection_name, top_k=10):
    emb = get_embedder()
    client = get_qdrant()
    query_vec = emb.encode(query)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec.tolist(),
        limit=top_k,
        with_payload=True
    )
    # result
    output = []
    for hit in results:
        output.append({
            "score": round(hit.score, 4),
            "section": hit.payload.get("section", ""),
            "text": hit.payload.get("text", "")
        })
    return output

# generate answer
answer_prompt = """Context from research paper:
%s

Question: %s

Provide a detailed answer based on the context above:"""

def generate_answer(query, chunks):
    if not chunks:
        return "no relevant context found"
    print("generating answer...")
    # concat context
    context = ""
    for c in chunks:
        context += f"[{c['section']}] {c['text']}\n\n"
    full_prompt = answer_prompt % (context, query)
    gemini = get_gemini()
    def call_api():
        resp = gemini.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return resp.text if resp and resp.text else "failed to generate answer"
    
    answer = retry(call_api, max_tries=2)
    return answer.strip()

# pipeline
"""process pdf pipeline"""
def process_pdf(pdf_path, collection_name, force=False):
    # check if already processed
    if not force and check_collection_exists(collection_name):
        count = get_collection_count(collection_name)
        if count > 0:
            print(f"collection '{collection_name}': {count} vectors -> skip")
            return False
    print(f"processing: {pdf_path}")
    # step 1 -> convert pdf
    md_path = convert_pdf(pdf_path)
    # step 2 -> read markdown
    md_content = read_markdown(md_path)
    print(f"loaded markdown: {len(md_content)} chars")
    # step 3 -> extract info
    paper_data = extract_info(md_content)
    # step 4 -> create chunks
    chunks = create_chunks(paper_data)
    # step 5 -> create collection
    create_collection(collection_name, recreate=force)
    # step 6: upload
    upload_chunks(chunks, collection_name)
    print(f"\nprocess done: {collection_name}")
    return True

"""ask question pipeline"""
def ask(question, collection_name):
    if not check_collection_exists(collection_name):
        raise Exception(f"collection not exists: {collection_name}")
    print(f"\nquery: {question}")
    # search
    chunks = search(question, collection_name)
    # generate answer
    answer = generate_answer(question, chunks)
    return answer, chunks

# get data evaluate rag
def eval_rag(collection_name, eval_csv, output_file):
    df = pd.read_csv(eval_csv)
    results = []
    last_call = 0
    interval = 8
    for _, row in df.iterrows():
        query = row['query']
        answer_true = row.get('answer_true', '')
        wait = interval - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        response, chunks = ask(query, collection_name)
        last_call = time.time()
        retrieved_docs = [c['text'] for c in chunks]
        results.append({
            "query": query,
            "answer_true": answer_true,
            "response": response,
            "retrieved_docs": json.dumps(retrieved_docs, ensure_ascii=False)
        })
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\nsaved: {output_file}")

# main
def main():  
    pdf_file = "abc.pdf"
    collection_name = Path(pdf_file).stem
    # check api key
    if not geminiApi:
        print("error: set geminiApiKey in file .env")
        return 1
    if not Path(dolphinModelPath).exists():
        print(f"error: not found dolphin model: {dolphinModelPath}")
        return 1
    print("rag pipeline starting")
    # process document
    try:
        process_pdf(pdf_file, collection_name, force=False)
    except Exception as e:
        print(f"error processing: {e}")
        return 1 
    
    eval_rag(collection_name, eval_csv="query_data.csv", output_file="eval_data.csv")
    
    # # interactive loop
    # print("Enter your question (type 'exit' to quit, 'reprocess' to process again)")
    # while True:
    #     try:
    #         question = input("\nQuestion: ").strip()
    #     except (EOFError, KeyboardInterrupt):
    #         print("\nExit.")
    #         break
    #     if question.lower() in ["exit", "quit", "q"]:
    #         print("Exit.")
    #         break
    #     if question.lower() == "reprocess":
    #         try:
    #             process_pdf(pdf_file, collection_name, force=True)
    #             print("Reprocessed.")
    #         except Exception as e:
    #             print(f"error: {e}")
    #         continue
    #     if not question:
    #         continue
    #     try:
    #         answer, chunks = ask(question, collection_name)
    #         print("\n")
    #         print("\nAnswer:")
    #         print(answer)
    #         print("\n")
    #         print("\nRelated paragraphs:")
    #         print("\n")
    #         for i, c in enumerate(chunks, 1):
    #             print(f"\n{i}. [{c['section']}] (score: {c['score']})")
    #             # get preview
    #             txt = c['text']
    #             if len(txt) > 200:
    #                 txt = txt[:200] + "..."
    #             print(f"{txt}")
    #     except Exception as e:
    #         print(f"error: {e}")
    # print("\n")
    # print("\nBye!")
    # return 0

if __name__ == "__main__":
    exit(main())