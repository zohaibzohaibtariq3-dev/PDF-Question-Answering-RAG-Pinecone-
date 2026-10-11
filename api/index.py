
import os 
from functools import lru_cache 
 
from fastapi import FastAPI, File, HTTPException, UploadFile 
from pydantic import BaseModel, Field 
 
from rag.config import get_settings 
from rag.pipeline import RAGPipeline 
 
app = FastAPI(title="PDF RAG API") 
 
 
@lru_cache(maxsize=1) 
def get_pipeline(): 
    settings = get_settings() 
    return RAGPipeline(settings) 
 
 
class AskRequest(BaseModel): 
    question: str = Field(min_length=1, max_length=5000) 
    namespaces: list[str] = Field(min_length=1, max_length=50) 
    top_k: int = Field(default=4, ge=1, le=10) 
    threshold: float = Field(default=0.30, ge=0, le=1) 
 
 
@app.get("/") 
def health(): 
    return {"status": "ok", "message": "PDF RAG API is running"} 
 
 
@app.get("/api/documents") 
def list_documents(): 
    try: 
        return {"documents": get_pipeline().list_documents()} 
    except Exception: 
        raise HTTPException(status_code=500, detail="Could not list documents") 
 
 
@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    try:
        result = get_pipeline().ingest(data, file.filename)
        return {
            "namespace": result.namespace,
            "filename": result.doc_name,
            "pages": result.pages,
            "chunks": result.chunks,
            "seconds": result.seconds,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ask")
def ask_question(request: AskRequest):
    try:
        answer = get_pipeline().ask(
            question=request.question.strip(),
            namespaces=request.namespaces,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        return {
            "question": answer.question,
            "answer": answer.text,
            "answered": answer.answered,
            "confidence_pct": answer.confidence_pct,
            "confidence_label": answer.confidence_label,
            "sources": [
                {
                    "text": source.text,
                    "score": source.score,
                }
                for source in answer.sources
            ],
        }

    except Exception as exc:
        print(f"Ask endpoint failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Ask failed: {type(exc).__name__}: {exc}",
        )


@app.delete("/api/documents/{namespace}")
def delete_document(namespace: str):
    try:
        get_pipeline().delete_document(namespace)
        return {"deleted": namespace}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not delete document")

