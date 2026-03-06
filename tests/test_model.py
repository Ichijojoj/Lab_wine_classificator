import pytest
import sys

sys.path.append('src')
from model import WineQualityModel
import joblib
import os


class TestWineQualityModel:

    @pytest.fixture
    def model(self):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=11, random_state=42)
        test_model = RandomForestClassifier(n_estimators=10, random_state=42)
        test_model.fit(X, y)
        joblib.dump(test_model, 'test_model.pkl')

        return WineQualityModel(model_path='test_model.pkl')

    def test_predict_returns_dict(self, model):
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

    def test_predict_quality_values(self, model):
        sample_input = {
            'fixed acidity': 7.0, 'volatile acidity': 0.27,
            'citric acid': 0.36, 'residual sugar': 20.7,
            'chlorides': 0.045, 'free sulfur dioxide': 45,
            'total sulfur dioxide': 170, 'density': 1.001,
            'pH': 3.0, 'sulphates': 0.45, 'alcohol': 8.8
        }
        result = model.predict(sample_input)
        assert result['quality'] in ['good', 'bad']
        assert 0 <= result['probability'] <= 1


def teardown_module(module):
    if os.path.exists('test_model.pkl'):
        os.remove('test_model.pkl')