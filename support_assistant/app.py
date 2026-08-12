from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_support_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MOCK_LLM = os.getenv("MOCK_LLM", "1").strip() != "0"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

POLICY_KEYWORDS = (
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float
    response: Dict[str, Any]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SupportResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0, le=1)


print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

_documents: List[str] = []
_ids: List[str] = []
for path in sorted(DOCS_DIR.glob("doc_*.txt")):
    _ids.append(path.stem)
    _documents.append(path.read_text(encoding="utf-8"))

_embeddings = embedder.encode(_documents, normalize_embeddings=True).tolist()
collection.add(ids=_ids, documents=_documents, embeddings=_embeddings)


PROMPT_TEMPLATE = """Role:
You are Zepto's support-policy assistant.

Context:
Use ONLY the supplied retrieved policy context. If the answer is not present in that
context, say that the provided policy context does not contain the answer.

Task:
Answer the customer's question accurately and briefly using only the retrieved context.

Format:
Return a concise customer-facing answer, followed by the source document/chunk IDs
when they are available.

Length:
Keep the answer to 2-4 sentences unless a short list is clearer.

Negative constraint:
Do not answer using information that is not present in the provided context.

Few-shot example:
Question: "How long can I report a damaged item?"
Context: "doc_06 — ... customers must report it within 24 hours of delivery ..."
Answer: "Damaged, spoiled, or missing items must be reported within 24 hours of delivery."
"""


def _real_llm(messages: List[Dict[str, str]]) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is required when MOCK_LLM=0.")
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY when MOCK_LLM=0.")
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    lowered = query.lower()

    if MOCK_LLM:
        intent = (
            "policy_question"
            if any(keyword in lowered for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        prompt = (
            "Classify this query as exactly one label: policy_question or "
            "general_question. Return only the label.\nQuery: " + query
        )
        label = _real_llm([
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": prompt},
        ]).strip().lower()
        intent = "policy_question" if "policy_question" in label else "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    q_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=3,
        include=["documents", "distances"],
    )

    retrieved: List[Dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc_id, document, distance in zip(ids, documents, distances):
        retrieved.append({
            "id": doc_id,
            "document": document,
            "distance": float(distance),
        })

    top = retrieved[0]
    top_snippet = top["document"][:200]

    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top_snippet}"
    else:
        prompt = (
            PROMPT_TEMPLATE
            + "\nRetrieved context:\n"
            + "\n\n".join(f"[{item['id']}] {item['document']}" for item in retrieved)
            + f"\n\nCustomer question: {query}"
        )
        answer = _real_llm([
            {"role": "system", "content": "Follow the structured support prompt exactly."},
            {"role": "user", "content": prompt},
        ])

    return {
        "retrieved": retrieved,
        "answer": answer,
        "sources": [item["id"] for item in retrieved],
        "confidence": 1.0 if MOCK_LLM else max(0.0, min(1.0, 1.0 - top["distance"])),
    }


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = _real_llm([
            {"role": "system", "content": "Answer briefly and helpfully."},
            {"role": "user", "content": state["query"]},
        ])

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }


def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.add_edge(START, "classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
graph = builder.compile()


def answer_query(query: str) -> SupportResponse:
    state = graph.invoke({"query": query})
    payload = SupportResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        confidence=float(state.get("confidence", 1.0)),
    )
    return payload


app = FastAPI(title="Zepto Support Assistant", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mock_llm": MOCK_LLM,
        "collection": COLLECTION_NAME,
        "documents_indexed": collection.count(),
    }


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest) -> SupportResponse:
    return answer_query(request.query)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
