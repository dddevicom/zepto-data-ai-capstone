# Module 3 — Support Assistant

A fully offline, deterministic Zepto support assistant using a local policy corpus,
`all-MiniLM-L6-v2` embeddings, ChromaDB retrieval, a LangGraph StateGraph router,
Pydantic-validated responses, and a FastAPI `/ask` endpoint.

## Required baseline

The graded path uses `MOCK_LLM=1` (the default). No LLM API key is required and
the application makes no network call to an LLM provider.

Embeddings are generated locally with `sentence-transformers` and persisted in a
local ChromaDB collection. The first run may download `all-MiniLM-L6-v2` if it is
not already cached on the machine.

## Install and run

```powershell
cd support_assistant
python -m pip install -r requirements.txt
$env:MOCK_LLM="1"
python app.py
```

The API listens on `http://localhost:7860`.

## Example API calls

Policy question (retrieval path):

```powershell
curl.exe -X POST http://localhost:7860/ask `
  -H "Content-Type: application/json" `
  -d '{"query":"What is the delivery fee for orders below INR 149?"}'
```

General question (direct path):

```powershell
curl.exe -X POST http://localhost:7860/ask `
  -H "Content-Type: application/json" `
  -d '{"query":"Who is the president of France?"}'
```

Expected mock response shape:

```json
{
  "answer": "...",
  "sources": ["doc_01", "doc_05", "doc_..."],
  "confidence": 1.0
}
```

General questions return `sources: []`.

## Architecture

```text
8 policy docs
    |
    v
ingestion -> per-document chunks -> all-MiniLM-L6-v2 -> ChromaDB
                                                     |
user query -> classify_intent -----------------------+
                  |                                   |
        policy_question                       general_question
                  |                                   |
                  v                                   v
       retrieve_and_answer                       direct_answer
                  |                                   |
                  +---------------+-------------------+
                                  v
                         Pydantic validation
                                  |
                                  v
                            FastAPI /ask
```

### Component map

- **Ingestion/chunking:** `docs/` and startup indexing in `app.py`.
- **Embedding:** `SentenceTransformer("all-MiniLM-L6-v2")`.
- **Vector store/retrieval:** ChromaDB collection `zepto_support_policies`, cosine similarity, top 3.
- **Routing:** LangGraph `StateGraph` with `classify_intent`, `retrieve_and_answer`, and `direct_answer`.
- **Generation:** `MOCK_LLM=1` uses deterministic code. `MOCK_LLM=0` optionally uses a Groq-compatible OpenAI client.
- **Output contract:** Pydantic `SupportResponse` with `answer`, `sources`, and `confidence`.
- **API:** FastAPI `POST /ask`.

## Prompt template

`PROMPT_TEMPLATE` in `app.py` contains the five requested skeleton components:
Role, Context, Task, Format, and Length, plus an explicit negative constraint and
a few-shot example as actual prompt text.

## MOCK_LLM behavior

With the default `MOCK_LLM=1`:
- `classify_intent` uses the required keyword heuristic.
- `retrieve_and_answer` performs real local embedding + Chroma retrieval and returns
  `Based on the retrieved context: ...`.
- `direct_answer` returns a fixed deterministic string.
- Pydantic response values are deterministic and confidence is `1.0`.

With `MOCK_LLM=0`, the optional generation/classification path uses the configured
Groq-compatible OpenAI endpoint. Set `GROQ_API_KEY` and optionally `LLM_MODEL`.
This extension is not required for grading.

## Docker

Build:

```powershell
docker build -t zepto-support-assistant .
```

Run:

```powershell
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

Then call `/ask` as shown above. The Dockerfile is intentionally baseline-first:
no external LLM API key is required.

## Example transcripts

See `examples/mock_transcripts.txt` for two recorded default-mode examples:
one retrieval query and one direct query.

## Reproducibility

The corpus is committed as plain text under `docs/`. ChromaDB is rebuilt from those
documents when the application starts, so the vector database is reproducible and
does not need to be committed.
