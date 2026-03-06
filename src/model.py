import joblib
import pandas as pd
import numpy as np
import os

class WineQualityModel:
    def __init__(self, model_path='models/wine_model.pkl', 
                 scaler_path='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = [
            'fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
    def predict(self, features_dict):

        for feature in self.feature_names:
            if feature not in features_dict:
                raise ValueError(f"Missing feature: {feature}")

        input_df = pd.DataFrame([features_dict], columns=self.feature_names)
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        return {
            'quality': 'good' if prediction == 1 else 'bad',
            'quality_class': int(prediction),
            'probability': float(max(probabilities)),
            'probabilities': {
                'bad': float(probabilities[0]),
                'good': float(probabilities[1])
            }
        }
    
    def predict_batch(self, features_list):
        results = []
        for features_dict in features_list:
            results.append(self.predict(features_dict))
        return results

if __name__ == "__main__":
    model = WineQualityModel()
    test_input = {
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
    
    result = model.predict(test_input)
    print("Result:", result)