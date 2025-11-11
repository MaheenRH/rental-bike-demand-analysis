import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

def prepare_data(df: pd.DataFrame):
    """Prepare features and target for regression models."""
    X = df[['temp', 'hum', 'windspeed', 'hr']]
    y = df['cnt']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_and_save_models(X_train, X_test, y_train, y_test):
    """Train Random Forest and Gradient Boosting models."""
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[name] = {
            "R2": round(r2_score(y_test, preds), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "MAE": round(mean_absolute_error(y_test, preds), 2)
        }
        joblib.dump(model, f"{name}_model.pkl")
        print(f"{name} trained → R²={metrics[name]['R2']}, RMSE={metrics[name]['RMSE']}")
    return metrics
