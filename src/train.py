
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,  roc_auc_score
import joblib
import configparser
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from data_preprocessing import load_data, preprocess_data, split_and_scale

def train_model(config_path='configs/config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    print("\nЗагрузка данных...")
    df = load_data(config_path)
    print(f"Загружено {len(df)} образцов")

    print("\nПредобработка...")
    X, y, feature_columns = preprocess_data(df)
    print(f" Признаков: {len(feature_columns)}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, 
        test_size=float(config['MODEL']['test_size']),
        random_state=int(config['MODEL']['random_state'])
    )
    print(f" Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n Обучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=int(config['MODEL']['n_estimators']),
        max_depth=int(config['MODEL']['max_depth']),
        min_samples_split=int(config['MODEL']['min_samples_split']),
        min_samples_leaf=int(config['MODEL']['min_samples_leaf']),
        random_state=int(config['MODEL']['random_state']),
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    print(" Модель обучена!")
    
    # Оценка
    print("\n Оценка модели...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f" ROC-AUC: {roc_auc:.4f}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))

    print("\n Сохранение модели...")
    model_path = config['MODEL']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, config['MODEL']['scaler_path'])

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_names': feature_columns,
        'feature_importance': dict(zip(feature_columns, model.feature_importances_))
    }
    joblib.dump(metrics, config['MODEL']['metrics_path'])
    
    print(f" Модель сохранена: {model_path}")
    print(f" Scaler сохранён: {config['MODEL']['scaler_path']}")
    print(f" Метрики сохранены: {config['MODEL']['metrics_path']}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = train_model()