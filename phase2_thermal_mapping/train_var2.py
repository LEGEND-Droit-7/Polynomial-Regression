import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "BT2024203_ML_Dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "BT2024203_train_var2.csv")
TEST_PATH = os.path.join(DATA_DIR, "BT2024203_test_var2.csv")
PRED_PATH = os.path.join(BASE_DIR, "BT2024203_pred_var2.csv")
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Loading Phase 2 Dataset (var2)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X = train_df.drop(columns=['y'])
    y = train_df['y']
    X_test = test_df.drop(columns=['y']) if 'y' in test_df.columns else test_df
    
    degrees_to_test = [1, 2, 4, 6, 8, 9, 10]
    cv_mses = []
    cv_r2s = []
    
    best_degree = 1
    best_score = -float('inf')
    best_alpha = None
    
    print("\n--- Sweeping Degrees with RidgeCV ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas_to_test = np.logspace(-4, 2, 50)
    
    for degree in degrees_to_test:
        print(f"Evaluating Degree {degree}...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        scaler = StandardScaler()
        
        X_poly = poly.fit_transform(X)
        X_scaled = scaler.fit_transform(X_poly)
        
        ridge_cv = RidgeCV(alphas=alphas_to_test, cv=kf)
        ridge_cv.fit(X_scaled, y)
        
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=ridge_cv.alpha_))
        ])
        
        scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2', n_jobs=-1)
        scores_mse = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        
        mean_r2 = np.mean(scores_r2)
        mean_mse = np.mean(scores_mse)
        
        cv_r2s.append(mean_r2)
        cv_mses.append(mean_mse)
        
        print(f" -> Best Alpha: {ridge_cv.alpha_:.4f} | Mean R2: {mean_r2:.4f} | Mean MSE: {mean_mse:.4f}")
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_degree = degree
            best_alpha = ridge_cv.alpha_
            
    print(f"\n=> Optimal Degree: {best_degree} with Ridge alpha={best_alpha:.4f}")
    
    # 1. Bias-Variance Plot
    plt.figure(figsize=(10, 5))
    plt.plot(degrees_to_test, cv_mses, marker='o', color='blue', label='CV MSE')
    plt.yscale('log')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Validation MSE (Log Scale)')
    plt.title('var2: Bias-Variance Curve vs Degree')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'var2_bias_variance.png'), bbox_inches='tight')
    plt.close()

    print("\n--- Training Final Model ---")
    final_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=best_alpha))
    ])
    final_pipeline.fit(X, y)
    
    # 2. Residuals Plot
    y_pred_cv = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        pipeline_cv = Pipeline([
            ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=best_alpha))
        ])
        pipeline_cv.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_cv[val_idx] = pipeline_cv.predict(X.iloc[val_idx])
        
    residuals = y - y_pred_cv
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction vs Actual
    axes[0].scatter(y_pred_cv, y, alpha=0.5)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[0].set_xlabel('Cross-Validated Prediction')
    axes[0].set_ylabel('Ground Truth Actuals')
    axes[0].set_title('var2: Actual vs Predicted')
    
    # Residuals Histogram
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_title('var2: Residual Distribution')
    
    plt.savefig(os.path.join(PLOT_DIR, 'var2_residuals.png'), bbox_inches='tight')
    plt.close()
    
    print("\n--- Generating Test Predictions ---")
    predictions = final_pipeline.predict(X_test)
    submission = pd.DataFrame({'y': predictions})
    submission.to_csv(PRED_PATH, index=False)
    print(f"Predictions saved to: {PRED_PATH}")
    print("Plots generated successfully!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
