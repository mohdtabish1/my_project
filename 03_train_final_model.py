"""
03_train_final_model.py
Trains the final model on entire dataset and saves it
Based on testing phase, uses Gradient Boosting with tuned hyperparameters
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. CREATE ARTIFACTS DIRECTORY
# ============================================================================
print("="*70)
print("SETTING UP ARTIFACTS DIRECTORY")
print("="*70)

artifacts_dir = 'artifacts'
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)
    print(f"Created artifacts directory: {artifacts_dir}")
else:
    print(f"Artifacts directory already exists: {artifacts_dir}")

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv('cars24-car-price-cleaned-new.csv')
print(f"Dataset shape: {df.shape}")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Separate features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

print(f"Original data shape - X: {X.shape}, y: {y.shape}")

# Remove outliers using IQR method for numerical features
def remove_outliers(X, y, columns=None):
    """Remove outliers using IQR method"""
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns
    
    indices_to_keep = np.ones(len(X), dtype=bool)
    
    for col in columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        indices_to_keep &= (X[col] >= lower_bound) & (X[col] <= upper_bound)
    
    return X[indices_to_keep], y[indices_to_keep]

# Remove outliers
X_clean, y_clean = remove_outliers(X, y)

# Also remove outliers from target variable
y_lower = y_clean.quantile(0.01)
y_upper = y_clean.quantile(0.99)
mask = (y_clean >= y_lower) & (y_clean <= y_upper)
X_clean = X_clean[mask]
y_clean = y_clean[mask]

print(f"After outlier removal - X: {X_clean.shape}, y: {y_clean.shape}")
print(f"Rows removed: {len(X) - len(X_clean)}")

# ============================================================================
# 4. TRAIN FINAL MODEL ON ENTIRE DATASET
# ============================================================================
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70)

# Best hyperparameters identified from testing phase
# Using Gradient Boosting with tuned parameters
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42,
    subsample=0.8,
    verbose=1
)

print("\nTraining Gradient Boosting Regressor...")
print("Hyperparameters:")
print(f"  n_estimators: {model.n_estimators}")
print(f"  learning_rate: {model.learning_rate}")
print(f"  max_depth: {model.max_depth}")
print(f"  min_samples_split: {model.min_samples_split}")

model.fit(X_clean, y_clean)
print("\nModel training completed successfully!")

# ============================================================================
# 5. MODEL EVALUATION ON FULL DATA
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_pred_train = model.predict(X_clean)
r2_train = r2_score(y_clean, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_clean, y_pred_train))
mae_train = mean_absolute_error(y_clean, y_pred_train)

print(f"\nTraining Set Performance:")
print(f"  R² Score: {r2_train:.4f}")
print(f"  RMSE: {rmse_train:.4f}")
print(f"  MAE: {mae_train:.4f}")

# ============================================================================
# 6. SAVE MODEL
# ============================================================================
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"car_price_model_{timestamp}.pkl"
model_path = os.path.join(artifacts_dir, model_name)

joblib.dump(model, model_path)
print(f"Model saved: {model_path}")

# Also save a generic 'latest' model
latest_model_path = os.path.join(artifacts_dir, 'car_price_model_latest.pkl')
joblib.dump(model, latest_model_path)
print(f"Latest model saved: {latest_model_path}")

# ============================================================================
# 7. SAVE FEATURE NAMES AND PREPROCESSING INFO
# ============================================================================
print("\n" + "="*70)
print("SAVING PREPROCESSING ARTIFACTS")
print("="*70)

# Save feature names
feature_info = {
    'features': X.columns.tolist(),
    'n_features': len(X.columns),
    'target': 'selling_price'
}

import json
feature_info_path = os.path.join(artifacts_dir, 'feature_info.json')
with open(feature_info_path, 'w') as f:
    json.dump(feature_info, f, indent=4)
print(f"Feature info saved: {feature_info_path}")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Top 15)")
print("="*70)

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + feature_importance_df.head(15).to_string(index=False))

# Save feature importance
importance_path = os.path.join(artifacts_dir, 'feature_importance.csv')
feature_importance_df.to_csv(importance_path, index=False)
print(f"\nFeature importance saved: {importance_path}")

# ============================================================================
# 9. MODEL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL MODEL SUMMARY")
print("="*70)

summary = {
    'model_name': 'Gradient Boosting Regressor',
    'timestamp': timestamp,
    'training_samples': len(X_clean),
    'features': len(X.columns),
    'r2_score': float(r2_train),
    'rmse': float(rmse_train),
    'mae': float(mae_train),
    'hyperparameters': {
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'subsample': model.subsample
    }
}

summary_path = os.path.join(artifacts_dir, 'model_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"\nModel summary saved: {summary_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nArtifacts Location: {os.path.abspath(artifacts_dir)}")
print(f"\nGenerated Files:")
print(f"  - {model_name} (Timestamped model)")
print(f"  - car_price_model_latest.pkl (Latest model)")
print(f"  - feature_info.json (Feature metadata)")
print(f"  - feature_importance.csv (Feature importance)")
print(f"  - model_summary.json (Model summary)")
print("\n" + "="*70)
