import pytest
from fastapi.testclient import TestClient
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Тесты /health"""

    def test_health_status(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_health_returns_json(self):
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestPredictEndpoint:
    """Тесты  /predict"""

    @pytest.fixture
    def valid_wine_input(self):
        return {
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

    def test_predict_missing_field(self):
        """Тест: ошибка при отсутствии поля"""
        response = client.post("/predict", json={"fixed_acidity": 7.0})
        assert response.status_code == 422

    def test_predict_valid_input(self, valid_wine_input):
        """Тест: успешное предсказание с валидными данными"""
        response = client.post("/predict", json=valid_wine_input)

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "quality" in data
            assert data["quality"] in ["good", "bad"]
            assert "probability" in data
            assert 0 <= data["probability"] <= 1

    def test_predict_invalid_type(self):
        """Тест: ошибка при неверном типе данных"""
        response = client.post("/predict", json={
            "fixed_acidity": "not_a_number",
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
        })
        assert response.status_code == 422


class TestFeaturesEndpoint:
    """Тесты для /features"""

    def test_features_returns_list(self):
        response = client.get("/features")

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "features" in data
            assert isinstance(data["features"], list)
            assert len(data["features"]) == 11


class TestRootEndpoint:
    """Тесты для главной страницы"""

    def test_root_returns_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Wine Quality" in data["message"]