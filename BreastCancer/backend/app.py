
import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data models for the simplified feature set
class PredictionInput(BaseModel):
    texture_worst: float
    radius_se: float
    symmetry_worst: float
    concave_points_mean: float
    concavity_worst: float
    area_se: float
    radius_worst: float
    area_worst: float
    concavity_mean: float
    concave_points_worst: float
    
    class Config:
        schema_extra = {
            "example": {
                "texture_worst": 17.33,
                "radius_se": 1.095,
                "symmetry_worst": 0.4601,
                "concave_points_mean": 0.1471,
                "concavity_worst": 0.7119,
                "area_se": 153.4,
                "radius_worst": 25.38,
                "area_worst": 2019.0,
                "concavity_mean": 0.3001,
                "concave_points_worst": 0.2654
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    diagnosis: str
    probability: float
    timestamp: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: float

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer diagnosis based on the most important cell nucleus characteristics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
metadata = None

# Load model function
def load_model():
    global model, metadata
    try:
        # Check if model exists
        model_path = os.path.join(os.path.dirname(__file__), "models", "logistic_regression_model.pkl")
        metadata_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Try to load metadata
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            logger.info("Model metadata loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load metadata: {str(e)}. Will proceed with only important features.")
            metadata = {"features": None}
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Prepare features function
def prepare_features(input_data):
    try:
        global metadata
        
        # If we have metadata about all features
        if metadata and metadata.get("features"):
            all_features = metadata.get("features")
            
            # Create a dictionary with all features initialized to 0
            feature_dict = {feature: 0.0 for feature in all_features}
            
            # Update with provided features
            for feature, value in input_data.items():
                # Handle naming differences
                if feature == "concave_points_mean":
                    feature_dict["concave points_mean"] = value
                elif feature == "concave_points_worst":
                    feature_dict["concave points_worst"] = value
                else:
                    feature_dict[feature] = value
            
            # Create DataFrame
            df = pd.DataFrame([feature_dict])
        else:
            # If no metadata, just use the input features
            df = pd.DataFrame([input_data])
            
            # Handle column name differences if needed
            if 'concave_points_mean' in df.columns:
                df['concave points_mean'] = df['concave_points_mean']
                df.drop('concave_points_mean', axis=1, inplace=True)
            
            if 'concave_points_worst' in df.columns:
                df['concave points_worst'] = df['concave_points_worst']
                df.drop('concave_points_worst', axis=1, inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise ValueError(f"Failed to prepare features: {str(e)}")

# Prediction function
def predict_cancer(input_data):
    global model
    try:
        # Load model if not loaded
        if model is None:
            model = load_model()
        
        # Prepare features
        features = prepare_features(input_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability
        probability = model.predict_proba(features)[0][1]  # Probability of class 1 (malignant)
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return int(prediction), float(probability)
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise RuntimeError(f"Failed to make prediction: {str(e)}")

# Load model at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting application and loading model...")
    try:
        load_model()
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        # Continue without failing - we'll try to load again when needed

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Welcome to the Breast Cancer Prediction API (Simplified Features)"}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    global model
    try:
        # Try to load model if not loaded
        if model is None:
            model = load_model()
        
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "model_loaded": False, "error": str(e), "timestamp": time.time()}
        )

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
        return JSONResponse(
            status_code=500, 
            content={"error": f"Prediction failed: {str(e)}"}
        )

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)