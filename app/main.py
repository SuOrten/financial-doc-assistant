from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import json
import redis
from app.ingest import ingest_pdf
from app.agent import ask_question, reset_history

app = FastAPI(title="Financial Document Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Redis connection - falls back gracefully if Redis not available
try:
    r = redis.Redis(host="redis", port=6379, db=0, socket_connect_timeout=2)
    r.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False

SESSION_ID = "default_session"

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Financial Document Assistant is running", "redis": REDIS_AVAILABLE}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunk_count = ingest_pdf(file_path)
    return {
        "message": f"Successfully processed {file.filename}",
        "chunks_indexed": chunk_count
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Load history from Redis if available
    if REDIS_AVAILABLE:
        history_data = r.get(SESSION_ID)
        if history_data:
            import sys
            sys.modules['app.agent'].chat_history = json.loads(history_data)

    answer = ask_question(request.question)

    # Save updated history to Redis
    if REDIS_AVAILABLE:
        import app.agent as agent_module
        r.setex(SESSION_ID, 3600, json.dumps(agent_module.chat_history))

    return QuestionResponse(answer=answer)

@app.post("/reset")
def reset():
    reset_history()
    if REDIS_AVAILABLE:
        r.delete(SESSION_ID)
    return {"message": "Conversation history cleared"}

@app.get("/health")
def health():
    return {"status": "healthy", "redis": REDIS_AVAILABLE}