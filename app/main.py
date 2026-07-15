from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from app.ingest import ingest_pdf
from app.agent import ask_question

app = FastAPI(title="Financial Document Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Financial Document Assistant is running"}

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
    
    answer = ask_question(request.question)
    return QuestionResponse(answer=answer)

@app.get("/health")
def health():
    return {"status": "healthy"}