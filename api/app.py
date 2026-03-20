from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import WineQualityModel

app = FastAPI(
    title="🍷 Wine Quality Prediction API",
    description="API для предсказания качества вина на основе химических характеристик",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WineFeatures(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45,
                "total_sulfur_dioxide": 170,
                "density": 1.001,
                "ph": 3.0,
                "sulphates": 0.45,
                "alcohol": 8.8
            }
        },
        protected_namespaces=()
    )


    fixed_acidity: float = Field(..., description="Fixed acidity")
    volatile_acidity: float = Field(..., description="Volatile acidity")
    citric_acid: float = Field(..., description="Citric acid")
    residual_sugar: float = Field(..., description="Residual sugar")
    chlorides: float = Field(..., description="Chlorides")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide")
    density: float = Field(..., description="Density")
    ph: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates")
    alcohol: float = Field(..., description="Alcohol")


class PredictionResponse(BaseModel):
    """Модель ответа предсказания"""
    quality: str = Field(..., description="Качество вина (good/bad)")
    quality_class: int = Field(..., description="Класс качества (0/1)")
    probability: float = Field(..., description="Вероятность предсказания")
    probabilities: dict = Field(..., description="Вероятности по классам")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    class Config:
        protected_namespaces = ()

class FeaturesResponse(BaseModel):
    """Модель ответа со списком признаков"""
    features: List[str]


class BatchPredictionRequest(BaseModel):
    """Модель для пакетного предсказания"""
    samples: List[WineFeatures]


class BatchPredictionResponse(BaseModel):
    """Модель ответа для пакетного предсказания"""
    predictions: List[PredictionResponse]


model: Optional[WineQualityModel] = None


@app.on_event("startup")
async def load_model():

    global model
    try:
        model_path = os.getenv('MODEL_PATH', 'models/wine_model.pkl')
        scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')

        model = WineQualityModel(model_path=model_path, scaler_path=scaler_path)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"⚠️ Warning: Model not loaded: {e}")
        model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке приложения"""
    global model
    model = None
    print("👋 Model unloaded")


@app.get("/", tags=["Main"])
async def root():
    """Главная страница API"""
    return {
        "message": "🍷 Wine Quality Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Проверка здоровья сервиса

    Возвращает статус сервиса и загружена ли модель
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: WineFeatures):

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        features_dict = {
            'fixed acidity': features.fixed_acidity,
            'volatile acidity': features.volatile_acidity,
            'citric acid': features.citric_acid,
            'residual sugar': features.residual_sugar,
            'chlorides': features.chlorides,
            'free sulfur dioxide': features.free_sulfur_dioxide,
            'total sulfur dioxide': features.total_sulfur_dioxide,
            'density': features.density,
            'pH': features.ph,
            'sulphates': features.sulphates,
            'alcohol': features.alcohol
        }

        result = model.predict(features_dict)

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        results = []
        for sample in request.samples:
            features_dict = {
                'fixed acidity': sample.fixed_acidity,
                'volatile acidity': sample.volatile_acidity,
                'citric acid': sample.citric_acid,
                'residual sugar': sample.residual_sugar,
                'chlorides': sample.chlorides,
                'free sulfur dioxide': sample.free_sulfur_dioxide,
                'total sulfur dioxide': sample.total_sulfur_dioxide,
                'density': sample.density,
                'pH': sample.ph,
                'sulphates': sample.sulphates,
                'alcohol': sample.alcohol
            }
            results.append(model.predict(features_dict))

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/features", response_model=FeaturesResponse, tags=["Info"])
async def get_features():

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return FeaturesResponse(features=model.feature_names)



@app.get("/metrics", tags=["Info"])
async def get_metrics():

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        import joblib
        metrics_path = os.getenv('METRICS_PATH', 'models/metrics.pkl')
        metrics = joblib.load(metrics_path)

        return {
            "accuracy": metrics.get('accuracy', 0),
            "roc_auc": metrics.get('roc_auc', 0),
            "train_size": metrics.get('train_size', 0),
            "test_size": metrics.get('test_size', 0),
            "feature_importance": metrics.get('feature_importance', {})
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load metrics: {str(e)}"
        )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        reload=True
    )