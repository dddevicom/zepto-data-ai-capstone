# Zepto Data AI Capstone

A complete three-module data and AI capstone project covering data engineering, analytics and machine learning, and a GenAI-powered Zepto support assistant.

---

## Repository Structure

```text
zepto-data-ai-capstone/
|
├── data_pipeline/
│   ├── README.md
│   ├── requirements.txt
│   ├── run_pipeline.py
│   └── ...
|
├── analytics/
│   ├── README.md
│   ├── requirements.txt
│   ├── analytics_pipeline.py
│   ├── reload_model.py
│   └── ...
|
├── support_assistant/
│   ├── README.md
│   ├── requirements.txt
│   ├── app.py
│   ├── Dockerfile
│   ├── docs/
│   └── examples/
|
├── MODULE_1_ACCEPTANCE_CHECKLIST.md
├── MODULE_2_ACCEPTANCE_CHECKLIST.md
├── MODULE_3_ACCEPTANCE_CHECKLIST.md
└── README.md

---

# Dependency and Setup Strategy

This project uses separate `requirements.txt` files for each module because each module has different Python dependencies.

Install the requirements for the module you want to run.

No paid service is required. Module 3 provides a deterministic `MOCK_LLM=1` baseline that does not require an external LLM API key.

---

# Module 1 - Data Pipeline

## Purpose

Module 1 covers data ingestion, cleaning, transformation, SQLite storage, SQL analysis, validation, and SQL JOIN versus pandas `merge()` comparison.

## Design Decisions

- Python is used for ingestion and transformation.
- SQLite is used for local relational storage.
- Data is cleaned and validated before database loading.
- SQL queries are used for relational analysis.
- SQL JOIN is compared with pandas `merge()`.
- Execution and validation results are stored in the outputs directory.

## Setup

```powershell
cd data_pipeline
python -m pip install -r requirements.txt
```
## Run End-to-End

```powershell
python analytics_pipeline.py
```

Verify the saved model:

```powershell
python reload_model.py
```

See `analytics/README.md` for detailed instructions.

---

# Module 3 - GenAI Support Assistant

## Purpose

Module 3 implements a Zepto support-policy assistant using local policy documents, embeddings, ChromaDB retrieval, LangGraph routing, Pydantic validation, and FastAPI.

## Design Decisions

- Eight policy documents are stored locally.
- `all-MiniLM-L6-v2` is used for local embeddings.
- ChromaDB provides local vector retrieval.
- LangGraph provides assistant routing.
- Pydantic validates API responses.
- FastAPI provides `/health` and `/ask`.
- `MOCK_LLM=1` provides the deterministic baseline.
- Docker provides reproducible deployment.

## Setup

```powershell
cd support_assistant
python -m pip install -r requirements.txt
```

Set deterministic mode:

```powershell
$env:MOCK_LLM="1"
```

## Run End-to-End

```powershell
python app.py
```

The API runs at:

```text
http://localhost:7860
```

### Health Check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/health"
```

### Policy Question

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the delivery fee for orders below INR 149?"}'
```

### General Question

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Who is the president of France?"}'
```

See `support_assistant/README.md` for detailed instructions.

---

# Validation and Testing

## Module 1

Validation includes data validation, SQLite validation, SQL analysis, and SQL JOIN versus pandas `merge()` comparison.

## Module 2

Validation includes profiling, missing-value analysis, correlation analysis, classification comparison, imbalance analysis, SMOTE comparison, hyperparameter tuning, regression analysis, residual analysis, and model reload verification.

## Module 3

Testing includes `/health`, policy-question `/ask`, general-question `/ask`, local embeddings, ChromaDB retrieval, and Docker execution.

---

# Git Workflow

The repository uses a feature-branch workflow.

The Git history contains a feature branch with multiple commits and a merge back into `main`.

All three modules are maintained in this single GitHub repository.

---

# Acceptance Checklists

- `MODULE_1_ACCEPTANCE_CHECKLIST.md`
- `MODULE_2_ACCEPTANCE_CHECKLIST.md`
- `MODULE_3_ACCEPTANCE_CHECKLIST.md`

---

# Submission

Submit this project as one public GitHub repository containing:

```text
/data_pipeline
/analytics
/support_assistant
README.md
```

Repository:

https://github.com/dddevicom/zepto-data-ai-capstone