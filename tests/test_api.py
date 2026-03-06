import pytest
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'

def test_predict_endpoint(client):
    data = {
        'fixed acidity': 7.0, 'volatile acidity': 0.27,
        'citric acid': 0.36, 'residual sugar': 20.7,
        'chlorides': 0.045, 'free sulfur dioxide': 45,
        'total sulfur dioxide': 170, 'density': 1.001,
        'pH': 3.0, 'sulphates': 0.45, 'alcohol': 8.8
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 200

def test_predict_missing_feature(client):
    data = {'fixed acidity': 7.0}
    response = client.post('/predict', json=data)
    assert response.status_code == 400