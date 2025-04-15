

# app/prediction.py
import pickle
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = os.environ.get("MODEL_PATH", "../models/logistic_regression_model.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "../models/model_metadata.pkl")

# Global model variables
_model = None
_metadata = None

def get_model():
    """
    Load the model from disk or return the cached model
    
    Returns:
        The trained model
    """
    global _model
    
    if _model is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    return _model

def get_metadata():
    """
    Load the model metadata from disk or return the cached metadata
    
    Returns:
        The model metadata
    """
    global _metadata
    
    if _metadata is None:
        try:
            logger.info(f"Loading metadata from {METADATA_PATH}")
            with open(METADATA_PATH, "rb") as f:
                _metadata = pickle.load(f)
            logger.info("Metadata loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading metadata: {str(e)}. Will proceed without metadata.")
            _metadata = {"features": None}
    
    return _metadata

def prepare_features(input_data: Dict[str, float]) -> pd.DataFrame:
    """
    Prepare features for prediction, filling in missing features with zeros
    
    Args:
        input_data: Dictionary containing only the important features
        
    Returns:
        DataFrame with all features required by the model
    """
    try:
        # Get metadata to know all required features
        metadata = get_metadata()
        all_features = metadata.get("features")
        
        if all_features is None:
            # If metadata is not available, just use the provided features
            logger.warning("No feature metadata available. Using only provided features.")
            df = pd.DataFrame([input_data])
        else:
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
        
        logger.debug(f"Prepared features with shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise ValueError(f"Failed to prepare features: {str(e)}")

def predict_cancer(input_data: Dict[str, float]) -> Tuple[int, float]:
    """
    Make a prediction using the trained model
    
    Args:
        input_data: Dictionary of important feature values
        
    Returns:
        Tuple of (prediction, probability)
    """
    try:
        # Get the model
        model = get_model()
        
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