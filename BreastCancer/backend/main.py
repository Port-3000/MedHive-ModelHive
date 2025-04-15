
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any
import logging
import time
from .models import PredictionInput, PredictionOutput, HealthResponse
from .prediction import get_model, predict_cancer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer diagnosis based on cell nucleus characteristics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting application and loading model...")
    get_model()
    logger.info("Model loaded successfully")

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Welcome to the Breast Cancer Prediction API"}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        # Check if model is loaded
        model = get_model()
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

# Prediction endpoint
@app.post("/breast-cancer-detection", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        logger.info("Received prediction request")
        
        # Convert input model to dictionary
        feature_dict = input_data.dict()
        
        # Make prediction
        prediction, probability = predict_cancer(feature_dict)
        
        logger.info(f"Prediction completed: {prediction}")
        
        return {
            "prediction": prediction,
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "probability": probability,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


