# UNFINISHED PROJECT (STILL IN PROGRESS)

# Financial Document Assistant

A RAG-based Q&A system for financial documents built with LangGraph and FastAPI.

## Architecture
- **LangGraph** agent with retrieve → generate pipeline
- **ChromaDB** vector store for semantic document retrieval
- **OpenAI** embeddings and GPT-4o-mini for answer generation
- **FastAPI** REST API with Swagger UI
- **Docker** containerization

## Features
- Upload PDF financial documents
- Ask natural language questions about the documents
- Answers grounded in document context with source citations
- Hallucination reduction through retrieval-based generation

## Endpoints
- `POST /upload` — Upload and index a PDF
- `POST /ask` — Ask a question about uploaded documents
- `GET /health` — Health check

## Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
