from flask import Flask, request, jsonify
from src.model import WineQualityModel
import os

app = Flask(__name__)
model = WineQualityModel(model_path=os.getenv('MODEL_PATH', 'models/wine_model.pkl'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = [
            'fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]

        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        result = model.predict(data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)