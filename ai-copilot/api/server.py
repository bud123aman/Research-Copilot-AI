import os
import shutil
from fastapi import FastAPI, UploadFile, File
from core.ingestion import load_universal_document, get_smart_nodes
from core.retrieval import setup_index_and_retriever, reranker
from core.generation import create_query_engine
from pydantic import BaseModel

app = FastAPI(title="AI Research Copilot API")

active_query_engine = None

class QueryRequest(BaseModel):
    question: str

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """Ingests ANY supported file type, processes it, and updates the RAG Engine."""
    global active_query_engine
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    docs = load_universal_document(temp_file_path)
    leaf_nodes = get_smart_nodes(docs)
    
    hybrid_retriever = setup_index_and_retriever(leaf_nodes)
    
    active_query_engine = create_query_engine(hybrid_retriever, reranker)
    
    os.remove(temp_file_path)
    return {"status": "success", "message": f"{file.filename} indexed successfully."}

@app.post("/query")
async def ask_question(req: QueryRequest):
    if not active_query_engine:
        return {"error": "System is empty. Please upload a document first."}
    
    response = active_query_engine.query(req.question)
    
    sources = [node.node.metadata.get("file_name", "Unknown") for node in response.source_nodes]
    return {"answer": str(response), "sources_used": list(set(sources))}