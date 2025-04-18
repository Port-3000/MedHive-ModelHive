

# Main application entry point for Disease Diagnosis System
# src/app.py

import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from contextlib import asynccontextmanager

# Import routers
from src.routers import diagnosis, feedback
from src.utils.logger import setup_logger
from src.config import settings
from src.services.vector_store import DiseaseVectorStore
from src.services.symptom_extractor import SymptomExtractor
from src.services.diagnosis_engine import DiagnosisEngine

# Setup logger
logger = setup_logger("app")

# Store app state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Initializes services on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Initializing application services...")
    
    # Initialize vector store
    app_state["vector_store"] = DiseaseVectorStore()
    
    # Initialize symptom extractor
    app_state["symptom_extractor"] = SymptomExtractor()
    
    # Initialize diagnosis engine
    app_state["diagnosis_engine"] = DiagnosisEngine(
        vector_store=app_state["vector_store"],
        symptom_extractor=app_state["symptom_extractor"],
    )
    
    logger.info("Application services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application services...")
    
    # Clean up vector store
    if "vector_store" in app_state:
        app_state["vector_store"].close()
    
    logger.info("Application services shut down successfully")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID and timing to each request"""
    request_id = f"{time.time():.0f}"
    request.state.request_id = request_id
    
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request_id
        
        # Log request details
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.4f}s"
        )
        
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id}
        )

# Include routers
app.include_router(diagnosis.router)
app.include_router(feedback.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "vector_store": "vector_store" in app_state,
            "symptom_extractor": "symptom_extractor" in app_state,
            "diagnosis_engine": "diagnosis_engine" in app_state
        }
    }

# Main entrypoint for running the application directly
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=port,
        reload=settings.DEBUG
    )