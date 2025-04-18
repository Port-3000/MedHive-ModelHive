

# FastAPI Backend for Disease Diagnosis Chatbot
# This module implements the API endpoints for the diagnosis system

import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from diagnosis_model import DiagnosisRAG, create_langchain_rag

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Disease Diagnosis API",
    description="API for diagnosing diseases based on symptoms using LLM and RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize diagnosis model
diagnosis_rag = None
langchain_rag = None

# Models for API request/response
class SymptomRequest(BaseModel):
    user_input: str = Field(..., description="Natural language description of symptoms")

class DiagnosisResponse(BaseModel):
    user_input: str
    extracted_symptoms: List[str]
    similar_diseases: List[Dict[str, Any]]
    diagnosis: str
    timestamp: str

class FeedbackRequest(BaseModel):
    session_id: str
    user_input: str
    diagnosis: str
    user_feedback: str
    correct_diagnosis: Optional[str] = None

# Dependency to get diagnosis model
def get_diagnosis_model():
    global diagnosis_rag
    if diagnosis_rag is None:
        diagnosis_rag = DiagnosisRAG()
    return diagnosis_rag

# Dependency to get LangChain RAG pipeline
def get_langchain_model():
    global langchain_rag
    if langchain_rag is None:
        langchain_rag = create_langchain_rag()
    return langchain_rag

# Background task to log feedback for model improvement
def log_feedback(feedback_data: Dict[str, Any]):
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Log feedback to a file
        with open("logs/feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        
        logger.info(f"Feedback logged successfully: {feedback_data['session_id']}")
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")

# Middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log request details
    logger.info(f"Request: {request.method} {request.url.path} - Processed in {process_time:.4f}s")
    
    return response

# Routes
@app.get("/")
def read_root():
    return {"message": "Disease Diagnosis API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose_disease(request: SymptomRequest, model: DiagnosisRAG = Depends(get_diagnosis_model)):
    """
    Diagnose diseases based on symptoms described in natural language
    """
    try:
        result = model.diagnose(request.user_input)
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during diagnosis: {str(e)}")

@app.post("/diagnose-langchain")
def diagnose_langchain(request: SymptomRequest, rag_chain = Depends(get_langchain_model)):
    """
    Alternative endpoint using LangChain RAG pipeline
    """
    try:
        diagnosis = rag_chain.invoke(request.user_input)
        return {"diagnosis": diagnosis}
    except Exception as e:
        logger.error(f"Error during LangChain diagnosis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during diagnosis: {str(e)}")

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit feedback on diagnosis for model improvement
    """
    feedback_data = feedback.dict()
    feedback_data["timestamp"] = datetime.now().isoformat()
    
    # Add task to log feedback
    background_tasks.add_task(log_feedback, feedback_data)
    
    return {"message": "Feedback received", "status": "success"}

# Shutdown event
@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown"""
    global diagnosis_rag
    if diagnosis_rag:
        diagnosis_rag.close()
        diagnosis_rag = None
    logger.info("Application shutting down, resources cleaned up")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)