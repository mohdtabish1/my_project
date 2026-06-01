"""
02_model_testing.py
Tests different algorithms with train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv('cars24-car-price-cleaned-new.csv')
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Separate features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

print(f"Target variable (selling_price) shape: {y.shape}")
print(f"Features shape: {X.shape}")

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

# Also remove outliers from target variable
X_clean, y_clean = remove_outliers(X, y)
y_lower = y_clean.quantile(0.01)
y_upper = y_clean.quantile(0.99)
mask = (y_clean >= y_lower) & (y_clean <= y_upper)
X_clean = X_clean[mask]
y_clean = y_clean[mask]

print(f"\nAfter outlier removal:")
print(f"  X shape: {X_clean.shape}")
print(f"  y shape: {y_clean.shape}")
print(f"  Rows removed: {len(X) - len(X_clean)}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train-Test ratio: {X_train.shape[0]/X_test.shape[0]:.2f}:1")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")
print(f"Training set scaled shape: {X_train_scaled.shape}")
print(f"Test set scaled shape: {X_test_scaled.shape}")

# ============================================================================
# 5. MODEL TESTING AND COMPARISON
# ============================================================================
print("\n" + "="*70)
print("TESTING DIFFERENT ALGORITHMS")
print("="*70)

# Dictionary to store models and their results
results = {}

# 1. Linear Regression
print("\n1. Testing Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
results['Linear Regression'] = {'model': lr, 'r2': r2_lr, 'rmse': rmse_lr, 'mae': mae_lr}
print(f"   R² Score: {r2_lr:.4f}")
print(f"   RMSE: {rmse_lr:.4f}")
print(f"   MAE: {mae_lr:.4f}")

# 2. Ridge Regression
print("\n2. Testing Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
results['Ridge Regression'] = {'model': ridge, 'r2': r2_ridge, 'rmse': rmse_ridge, 'mae': mae_ridge}
print(f"   R² Score: {r2_ridge:.4f}")
print(f"   RMSE: {rmse_ridge:.4f}")
print(f"   MAE: {mae_ridge:.4f}")

# 3. Lasso Regression
print("\n3. Testing Lasso Regression...")
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
results['Lasso Regression'] = {'model': lasso, 'r2': r2_lasso, 'rmse': rmse_lasso, 'mae': mae_lasso}
print(f"   R² Score: {r2_lasso:.4f}")
print(f"   RMSE: {rmse_lasso:.4f}")
print(f"   MAE: {mae_lasso:.4f}")

# 4. Random Forest
print("\n4. Testing Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # No scaling needed for tree-based models
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
results['Random Forest'] = {'model': rf, 'r2': r2_rf, 'rmse': rmse_rf, 'mae': mae_rf}
print(f"   R² Score: {r2_rf:.4f}")
print(f"   RMSE: {rmse_rf:.4f}")
print(f"   MAE: {mae_rf:.4f}")

# 5. Gradient Boosting
print("\n5. Testing Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb = mean_absolute_error(y_test, y_pred_gb)
results['Gradient Boosting'] = {'model': gb, 'r2': r2_gb, 'rmse': rmse_gb, 'mae': mae_gb}
print(f"   R² Score: {r2_gb:.4f}")
print(f"   RMSE: {rmse_gb:.4f}")
print(f"   MAE: {mae_gb:.4f}")

# 6. Support Vector Regression
print("\n6. Testing Support Vector Regression...")
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
results['SVR'] = {'model': svr, 'r2': r2_svr, 'rmse': rmse_svr, 'mae': mae_svr}
print(f"   R² Score: {r2_svr:.4f}")
print(f"   RMSE: {rmse_svr:.4f}")
print(f"   MAE: {mae_svr:.4f}")

# ============================================================================
# 6. MODEL COMPARISON AND RANKING
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[m]['r2'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAE': [results[m]['mae'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('R² Score', ascending=False)
print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name}")
print(f"R² Score: {comparison_df.iloc[0]['R² Score']:.4f}")
print(f"RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
print(f"MAE: {comparison_df.iloc[0]['MAE']:.4f}")
print(f"{'='*70}")

# ============================================================================
# 7. HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================
print("\n" + "="*70)
print("HYPERPARAMETER TUNING FOR BEST MODEL")
print("="*70)

if best_model_name == "Gradient Boosting":
    print("\nTuning Gradient Boosting hyperparameters...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 10]
    }
    
    gb_tuned = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb_tuned, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    y_pred_tuned = grid_search.predict(X_test)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    
    print(f"\nTuned Model Performance on Test Set:")
    print(f"   R² Score: {r2_tuned:.4f}")
    print(f"   RMSE: {rmse_tuned:.4f}")
    print(f"   MAE: {mae_tuned:.4f}")
    
    # Store the best tuned model
    best_final_model = grid_search.best_estimator_
    
elif best_model_name == "Random Forest":
    print("\nTuning Random Forest hyperparameters...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_tuned, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    y_pred_tuned = grid_search.predict(X_test)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    
    print(f"\nTuned Model Performance on Test Set:")
    print(f"   R² Score: {r2_tuned:.4f}")
    print(f"   RMSE: {rmse_tuned:.4f}")
    print(f"   MAE: {mae_tuned:.4f}")
    
    best_final_model = grid_search.best_estimator_
else:
    print(f"\nUsing best model ({best_model_name}) without additional tuning")
    best_final_model = results[best_model_name]['model']

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
if hasattr(best_final_model, 'feature_importances_'):
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Top 10)")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + feature_importance.head(10).to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TESTING PHASE COMPLETE")
print("="*70)
print(f"\nBest model identified: {best_model_name}")
print("Next step: Run 03_train_final_model.py to train on full dataset")
print("="*70)
