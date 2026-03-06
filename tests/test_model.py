import pytest
import sys
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Добавляем корень проекта в path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import WineQualityModel


class TestWineQualityModel:

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Фикстура для очистки тестовых файлов"""
        yield
        # Cleanup после каждого теста
        for f in ['test_model.pkl', 'test_scaler.pkl']:
            if os.path.exists(f):
                os.remove(f)

    @pytest.fixture
    def mock_model_and_scaler(self, tmp_path):
        """Создаёт тестовые модель и скалер во временной папке"""
        # Генерация тестовых данных
        X, y = make_classification(
            n_samples=100, n_features=11, n_redundant=0,
            n_informative=11, random_state=42
        )

        # Обучение скалера
        scaler = StandardScaler()
        scaler.fit(X)

        # Обучение модели
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Пути для сохранения
        model_path = tmp_path / "test_model.pkl"
        scaler_path = tmp_path / "test_scaler.pkl"

        # Сохранение
        joblib.dump(model, str(model_path))
        joblib.dump(scaler, str(scaler_path))

        return str(model_path), str(scaler_path)

    def test_predict_returns_dict(self, mock_model_and_scaler):
        """Тест: предсказание возвращает dict"""
        model_path, scaler_path = mock_model_and_scaler

        # Инициализация модели с тестовыми путями
        model = WineQualityModel(model_path=model_path, scaler_path=scaler_path)

        sample_input = {
            'fixed acidity': 7.0,
            'volatile acidity': 0.27,
            'citric acid': 0.36,
            'residual sugar': 20.7,
            'chlorides': 0.045,
            'free sulfur dioxide': 45,
            'total sulfur dioxide': 170,
            'density': 1.001,
            'pH': 3.0,
            'sulphates': 0.45,
            'alcohol': 8.8
        }

        result = model.predict(sample_input)

        assert isinstance(result, dict)
        assert 'quality' in result
        assert 'probability' in result
        assert result['quality'] in ['good', 'bad']
        assert 0 <= result['probability'] <= 1

    def test_predict_quality_values(self, mock_model_and_scaler):
        """Тест: проверка значений качества"""
        model_path, scaler_path = mock_model_and_scaler
        model = WineQualityModel(model_path=model_path, scaler_path=scaler_path)

        sample_input = {
            'fixed acidity': 7.0, 'volatile acidity': 0.27,
            'citric acid': 0.36, 'residual sugar': 20.7,
            'chlorides': 0.045, 'free sulfur dioxide': 45,
            'total sulfur dioxide': 170, 'density': 1.001,
            'pH': 3.0, 'sulphates': 0.45, 'alcohol': 8.8
        }

        result = model.predict(sample_input)

        assert result['quality'] in ['good', 'bad']
        assert result['quality_class'] in [0, 1]
        assert 'probabilities' in result
        assert 'bad' in result['probabilities']
        assert 'good' in result['probabilities']

    def test_predict_missing_feature_raises(self, mock_model_and_scaler):
        """Тест: ошибка при отсутствии признака"""
        model_path, scaler_path = mock_model_and_scaler
        model = WineQualityModel(model_path=model_path, scaler_path=scaler_path)

        incomplete_input = {'fixed acidity': 7.0}  # Только один признак

        with pytest.raises(ValueError, match="Missing feature"):
            model.predict(incomplete_input)

    def test_feature_names_count(self, mock_model_and_scaler):
        """Тест: проверка количества признаков"""
        model_path, scaler_path = mock_model_and_scaler
        model = WineQualityModel(model_path=model_path, scaler_path=scaler_path)

        assert len(model.feature_names) == 11
        assert 'alcohol' in model.feature_names
        assert 'pH' in model.feature_names