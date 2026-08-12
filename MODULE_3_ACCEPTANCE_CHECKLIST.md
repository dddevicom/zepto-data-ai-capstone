# Module 3 — Acceptance Checklist

- [x] 8 corpus documents included under `docs/`.
- [x] Local `all-MiniLM-L6-v2` embeddings and ChromaDB collection.
- [x] Structured prompt with Role, Context, Task, Format, Length, negative constraint, and few-shot example.
- [x] LangGraph StateGraph with TypedDict state and 3 named nodes.
- [x] Conditional routing to retrieval/direct answer.
- [x] Default `MOCK_LLM=1` deterministic path.
- [x] Pydantic response schema: answer/sources/confidence.
- [x] FastAPI POST `/ask`.
- [x] Two mock-mode example transcripts.
- [x] Dockerfile for local build/run.
- [x] README architecture and run instructions.
