import pytest
from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["timestamp"], float)

def test_predict_valid_input():
    """Test prediction with valid input data"""
    test_data = {
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
    
    response = client.post("/api/v1/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "diagnosis" in data
    assert "probability" in data
    assert "timestamp" in data
    assert data["diagnosis"] in ["Benign", "Malignant"]
    assert 0 <= data["probability"] <= 1
    assert isinstance(data["timestamp"], float)

def test_predict_invalid_input():
    """Test prediction with invalid input data"""
    invalid_data = {
        "texture_worst": "invalid",  # Should be float
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
    
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_predict_missing_fields():
    """Test prediction with missing required fields"""
    incomplete_data = {
        "texture_worst": 17.33,
        "radius_se": 1.095,
        # Missing other required fields
    }
    
    response = client.post("/api/v1/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error

def test_predict_edge_cases():
    """Test prediction with edge case values"""
    edge_case_data = {
        "texture_worst": 0.0,
        "radius_se": 0.0,
        "symmetry_worst": 0.0,
        "concave_points_mean": 0.0,
        "concavity_worst": 0.0,
        "area_se": 0.0,
        "radius_worst": 0.0,
        "area_worst": 0.0,
        "concavity_mean": 0.0,
        "concave_points_worst": 0.0
    }
    
    response = client.post("/api/v1/predict", json=edge_case_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "diagnosis" in data
    assert "probability" in data 