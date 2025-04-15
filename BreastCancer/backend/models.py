
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class PredictionInput(BaseModel):
    texture_worst: float = Field(..., description="Worst or largest mean value for standard deviation of gray-scale values")
    radius_se: float = Field(..., description="Standard error for the mean of distances from center to points on the perimeter")
    symmetry_worst: float = Field(..., description="Worst or largest mean value for mean symmetry of cell nucleus")
    concave_points_mean: float = Field(..., description="Mean for number of concave portions of the contour")
    concavity_worst: float = Field(..., description="Worst or largest mean value for mean of severity of concave portions of the contour")
    area_se: float = Field(..., description="Standard error for mean area of the core tumor")
    radius_worst: float = Field(..., description="Worst or largest mean value for distance from center to points on the perimeter")
    area_worst: float = Field(..., description="Worst or largest mean value for mean area of the core tumor")
    concavity_mean: float = Field(..., description="Mean of severity of concave portions of the contour")
    concave_points_worst: float = Field(..., description="Worst or largest mean value for number of concave portions of the contour")
    
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
    prediction: int = Field(..., description="Prediction (0 for benign, 1 for malignant)")
    diagnosis: str = Field(..., description="Diagnosis interpretation (Benign or Malignant)")
    probability: float = Field(..., description="Probability of malignancy")
    timestamp: float = Field(..., description="Timestamp of the prediction")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the service")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    timestamp: float = Field(..., description="Current timestamp")