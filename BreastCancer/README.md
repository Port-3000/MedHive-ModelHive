# Breast Cancer Prediction API - Simplified Features

Not: This is an ai generated readme proper docs needs to be updated

This project provides a machine learning pipeline for breast cancer prediction using logistic regression and a FastAPI backend that can be hosted on Huggingface Spaces. The API has been optimized to use only the most important features for prediction, making it more streamlined for UI implementations.

## Feature Optimization

Based on the feature importance analysis, we've identified the top 10 most important features for prediction:

| Feature | Coefficient | AbsCoef |
|---------|-------------|---------|
| texture_worst | 1.350606 | 1.350606 |
| radius_se | 1.268178 | 1.268178 |
| symmetry_worst | 1.208200 | 1.208200 |
| concave_points_mean | 1.119804 | 1.119804 |
| concavity_worst | 0.943053 | 0.943053 |
| area_se | 0.907186 | 0.907186 |
| radius_worst | 0.879840 | 0.879840 |
| area_worst | 0.841846 | 0.841846 |
| concavity_mean | 0.801458 | 0.801458 |
| concave_points_worst | 0.778217 | 0.778217 |

The API now only requires these features for prediction, significantly reducing input complexity while maintaining high prediction accuracy.

## Project Structure

```
breast-cancer-predictor/
├── data/
│   └── breast_cancer_data.csv
├── notebooks/
│   └── breast_cancer_model_development.ipynb
├── models/
│   ├── logistic_regression_model.pkl
│   └── model_metadata.pkl
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── prediction.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip for package installation

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-predictor.git
cd breast-cancer-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your breast cancer dataset in the `data/` directory.

### Training the Model

1. Run the Jupyter Notebook:
```bash
jupyter notebook notebooks/breast_cancer_model_development.ipynb
```

2. Follow the steps in the notebook to train the model and save it to disk.

### Running the API Locally

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

2. The API will be available at `http://localhost:8000`

3. Access the API documentation at `http://localhost:8000/docs`

## API Endpoints

- `GET /`: Root endpoint with welcome message
- `GET /health`: Health check endpoint
- `POST /breast-cancer-detection`: Prediction endpoint that accepts feature data

### Using the API

To make a prediction, send a POST request to `/breast-cancer-detection` with the required features:

```bash
curl -X 'POST' \
  'http://localhost:8000/breast-cancer-detection' \
  -H 'Content-Type: application/json' \
  -d '{
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
  }'
```

The response will contain the prediction:
```json
{
  "prediction": 1,
  "diagnosis": "Malignant",
  "probability": 0.987,
  "timestamp": 1617293847.123
}
```

## Deploying to Huggingface Spaces

1. Create a new Space on [Huggingface Spaces](https://huggingface.co/spaces)
2. Choose the "Gradio/FastAPI" template
3. Upload the project files with the simplified `app.py` file included in this repository
4. The API will be automatically deployed

## Implementation Details

The refactored API preserves the full model capabilities while only requiring input of the most important features:

1. When a prediction request is received with only the top 10 features, the API:
   - Fills in any other features required by the model with zeros
   - Ensures feature names match the expected format
   - Applies the same preprocessing pipeline
   - Makes a prediction using the full model

2. The architecture maintains compatibility with the original model while simplifying the interface.

## Extension Support

This simplified API remains fully compatible with planned extensions:
- MLFlow integration
- Federated learning systems
- Monitoring and logging enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.