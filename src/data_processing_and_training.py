import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def main():
    print("Starting data processing and training...")
    # Paths from project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    metadata_path = os.path.join(DATA_DIR, 'nasa_battery_dataset.csv')
    metadata = pd.read_csv(metadata_path)

    # Filter for discharge cycles which contain Capacity
    discharge_metadata = metadata[metadata['type'] == 'discharge'].copy()
    discharge_metadata['Capacity'] = pd.to_numeric(discharge_metadata['Capacity'], errors='coerce')
    discharge_metadata = discharge_metadata.dropna(subset=['Capacity'])

    features = []
    capacities = []
    print("Extracting features from raw sensor data...")
    for idx, row in discharge_metadata.iterrows():
        filename = row['filename']
        filepath = os.path.join(DATA_DIR, 'raw_sensor_data', filename)
        capacity = row['Capacity']
        if os.path.exists(filepath):
            try:
                sensor_df = pd.read_csv(filepath)
                # Feature engineering basics
                f_dict = {}
                f_dict['ambient_temperature'] = row['ambient_temperature']
                for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured']:
                    f_dict[col+'_mean'] = sensor_df[col].mean()
                    f_dict[col+'_max'] = sensor_df[col].max()
                    f_dict[col+'_min'] = sensor_df[col].min()
                features.append(f_dict)
                capacities.append(capacity)
            except Exception as e:
                pass

    X = pd.DataFrame(features)
    y = np.array(capacities)

    # Strict Pandas imputation
    print("Imputing missing values...")
    X = X.fillna(X.mean())

    # Scikit-Learn StandardScaler
    print("Splitting dataset and applying StandardScaler...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train 3 models
    print("Training models...")
    models = {
        'Multiple Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_rmse = float('inf')
    best_model_name = ""
    results = {}

    features_importance = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-Squared': r2}
        print(f"[{name}] RMSE: {rmse:.4f} | R-Squared: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name
            
            if hasattr(model, 'feature_importances_'):
                features_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                features_importance = model.coef_

    print(f"\nWinning model: {best_model_name} with RMSE: {best_rmse:.4f}")

    # Generating visualization data for Streamlit
    print("Compiling metadata for Streamlit charts...")
    
    # 1. Loss Graph (using XGBoost to simulate learning curve)
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)], verbose=False)
    results_xgb = xgb.evals_result()
    # Depending on metric name, normally uses rmse if objective is reg:squarederror
    metric_key = list(results_xgb['validation_0'].keys())[0]
    train_loss = results_xgb['validation_0'][metric_key]
    val_loss = results_xgb['validation_1'][metric_key]

    # 2. Degradation Curve
    # Simulate a lifecycle from high to low capacity for visualization
    indices = np.argsort(y_test)[::-1]
    deg_actual = y_test[indices]
    deg_pred = best_model.predict(X_test_scaled)[indices]

    # 3. Residuals
    best_preds = best_model.predict(X_test_scaled)
    residuals = y_test - best_preds

    training_metadata = {
        'feature_names': list(X.columns),
        'feature_importance': features_importance,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'deg_actual': deg_actual,
        'deg_pred': deg_pred,
        'y_test': y_test,
        'residuals': residuals,
        'model_evals': results,
        'scaler': scaler
    }

    print("Saving models to /models/...")
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    with open(os.path.join(MODELS_DIR, 'training_metadata.pkl'), 'wb') as f:
        pickle.dump(training_metadata, f)
        
    print("Done! Model and metadata serialized successfully.")

if __name__ == '__main__':
    main()
