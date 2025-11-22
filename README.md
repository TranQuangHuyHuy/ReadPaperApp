# RAG Internship Project

A **Retrieval-Augmented Generation (RAG)** system combining document image parsing with vector search and LLM-based Q&A.

## Overview

This project integrates several components:
- **Dolphin**: Document image parsing model (from ByteDance) for converting PDFs/images to structured text
- **Qdrant**: Vector database for semantic search over document chunks
- **Gemini API**: LLM for generating answers based on retrieved context
- **Sentence Transformers**: Embeddings for semantic similarity search
- **RAG Evaluation**: Hit@k metric to measure retrieval quality

## Architecture

```
PDF/Image Input
    ↓
[Dolphin] Document Parser → Structured Text
    ↓
[Sentence Transformer] Embedding Model → Vector Embeddings
    ↓
[Qdrant] Vector Database (store chunks + embeddings)
    ↓
Query Input
    ↓
[Semantic Search] Retrieve Top-k Relevant Chunks
    ↓
[Gemini LLM] Generate Answer (with retrieved context)
    ↓
Output Answer + Source Chunks
```

## Project Structure

```
.
├── app.py                      # Main RAG pipeline (Qdrant + Gemini)
├── rag_eval.py                 # Evaluation script (Hit@k metric)
├── demoo.py                    # Quick demo using Dolphin document parser
├── eval_data.csv               # Evaluation dataset (query, answer_true, response, retrieved_docs)
├── query_data.csv              # Query dataset
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (geminiApiKey, etc.)
├── abc.pdf                     # Sample PDF for testing
├── output/                     # Results folder
├── Dolphin/                    # Document parser submodule
│   ├── README.md              # Dolphin documentation
│   ├── hf_model/              # Pre-trained Dolphin model (download required)
│   ├── requirements.txt        # Dolphin dependencies
│   └── demo_page.py           # Dolphin page-level parsing
└── venv/                       # Virtual environment (created on setup)
```

## Setup

### 1. Clone and Install Dependencies

```powershell
# Activate virtual environment
python -m venv venv
& .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dolphin Model

The Dolphin document parser requires a pre-trained model:

```powershell
# Download from Hugging Face
huggingface-cli download ByteDance/Dolphin-1.5 --local-dir ./Dolphin/hf_model
```

Or manually:
```powershell
git lfs install
git clone https://huggingface.co/ByteDance/Dolphin-1.5 ./Dolphin/hf_model
```

### 3. Configure Environment Variables

Create or update `.env`:

```env
geminiApiKey=your_gemini_api_key_here
```

Get your Gemini API key: https://aistudio.google.com/app/apikey

### 4. Start Qdrant Vector Database

Run Qdrant locally (requires Docker):

```powershell
docker run -p 6333:6333 -p 6334:6334 `
  -e QDRANT_API_KEY="your_secret_key" `
  qdrant/qdrant:latest
```

Or use Qdrant cloud: https://qdrant.tech/

## Usage

### Parse a Document and Build Vector Store

```powershell
# Run the main RAG pipeline
python app.py
```

This will:
1. Parse documents using Dolphin
2. Generate embeddings using Sentence Transformers
3. Store in Qdrant
4. Accept user queries and retrieve relevant chunks
5. Generate answers using Gemini LLM

### Evaluate Retrieval Quality

```powershell
# Run evaluation on eval_data.csv
python rag_eval.py
```

Output:
```
Hit@k: 0.8750  # Percentage of queries where ground truth answer was in top-k retrieved docs
```

### Quick Demo: Document Parsing

```powershell
# Parse a PDF using Dolphin
python demoo.py
```

This processes `abc.pdf` and outputs:
- Structured markdown
- JSON metadata
- Recognized elements (text, tables, formulas)

## Configuration

### Key Variables in `app.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `geminiModelName` | `gemini-2.5-flash` | LLM model for generation |
| `qdrantHost` | `localhost` | Qdrant server host |
| `qdrantPort` | `6333` | Qdrant server port |
| `embeddingModel` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model |
| `embeddingDim` | `1024` | Embedding dimension |
| `dolphinModelPath` | `./Dolphin/hf_model` | Path to Dolphin model |
| `outputPath` | `./output` | Output directory |

### Evaluation Metric (Hit@k)

`rag_eval.py` computes:

$$\text{Hit@k} = \frac{\text{# queries where } \text{ground\_truth} \in \text{top-k retrieved docs}}{\text{total queries}}$$

**Normalization**: Text is lowercased and whitespace-collapsed before substring matching.

## Common Issues

### Issue: NumPy / TensorFlow Mismatch
**Error**: `module 'numpy' has no attribute 'dtypes'`

**Solution**:
```powershell
pip install "numpy<2.0"
# or
pip install --upgrade tensorflow
```

### Issue: Dolphin Model Not Found
**Error**: `FileNotFoundError: ./Dolphin/hf_model`

**Solution**: Download the model as shown in Setup step 2.

### Issue: Qdrant Connection Failed
**Error**: `Failed to connect to Qdrant at localhost:6333`

**Solution**: Start Qdrant (see Setup step 4) or update `qdrantHost` / `qdrantPort` in `app.py`.

### Issue: Gemini API Key Invalid
**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solution**: Ensure `.env` has a valid `geminiApiKey` and `load_dotenv()` is called before API usage.

## Example Workflow

```powershell
# 1. Activate environment
& .\venv\Scripts\Activate.ps1

# 2. Start Qdrant (in separate terminal)
docker run -p 6333:6333 qdrant/qdrant:latest

# 3. Parse a document and index it
python app.py
# → Prompts for PDF path, parses, embeds, stores in Qdrant

# 4. Ask a question (app.py will accept input)
# → Input: "What is the main topic?"
# → Output: Retrieved chunks + LLM-generated answer

# 5. Evaluate retrieval on test dataset
python rag_eval.py
# → Hit@k: 0.8750
```

## Data Format

### `eval_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `query` | string | User question |
| `answer_true` | string | Ground truth / expected answer |
| `response` | string | Model-generated response |
| `retrieved_docs` | JSON list | Retrieved document chunks (as JSON array) |

Example:
```csv
query,answer_true,response,retrieved_docs
"What is the title?","Solar Power Forecasting","The title is...",["[Section] ... Solar Power...", "[Section] ... Forecasting..."]
```

## Dependencies

- **google-generativeai**: Gemini API client
- **sentence-transformers**: Embeddings
- **qdrant-client**: Vector DB client
- **torch, transformers**: NLP models
- **pandas, numpy**: Data processing
- **opencv-python, Pillow**: Image handling
- **pypdfium2**: PDF reading

See `requirements.txt` for exact versions.

## References

- **Dolphin**: https://github.com/bytedance/Dolphin (document parsing)
- **Qdrant**: https://qdrant.tech/ (vector search)
- **Gemini API**: https://aistudio.google.com/
- **Sentence Transformers**: https://www.sbert.net/

## License

ByteDance Dolphin: MIT  
This workspace: Custom (eval scripts, RAG integration)

---

**Questions?** Check Dolphin's `README.md` for detailed document parsing options, or refer to the source code comments in `app.py`.
