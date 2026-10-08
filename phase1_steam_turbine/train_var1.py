import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "BT2024203_ML_Dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "BT2024203_train_var1.csv")
TEST_PATH = os.path.join(DATA_DIR, "BT2024203_test_var1.csv")
PRED_PATH = os.path.join(BASE_DIR, "BT2024203_pred_var1.csv")
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("Loading Phase 1 Dataset (var1)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X = train_df.drop(columns=['y'])
    y = train_df['y']
    X_test = test_df.drop(columns=['y']) if 'y' in test_df.columns else test_df
    
    degrees_to_test = [1, 2, 3, 4, 5, 6]
    cv_mses = []
    cv_r2s = []
    
    best_degree = 1
    best_score = -float('inf')
    best_alpha = None
    
    print("\n--- Sweeping Degrees with LassoCV ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas_to_test = np.logspace(-4, 0, 50)
    
    for degree in degrees_to_test:
        print(f"Evaluating Degree {degree}...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        scaler = StandardScaler()
        
        # We transform manually for CV to get the best alpha directly from LassoCV
        X_poly = poly.fit_transform(X)
        X_scaled = scaler.fit_transform(X_poly)
        
        # LassoCV automatically does cross-validation to find the best alpha
        lasso_cv = LassoCV(alphas=alphas_to_test, cv=kf, max_iter=20000, n_jobs=-1, tol=1e-3)
        lasso_cv.fit(X_scaled, y)
        
        # Calculate CV R2 and MSE manually to be sure
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=lasso_cv.alpha_, max_iter=20000, tol=1e-3))
        ])
        
        scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2', n_jobs=-1)
        scores_mse = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        
        mean_r2 = np.mean(scores_r2)
        mean_mse = np.mean(scores_mse)
        
        cv_r2s.append(mean_r2)
        cv_mses.append(mean_mse)
        
        print(f" -> Best Alpha: {lasso_cv.alpha_:.4f} | Mean R2: {mean_r2:.4f} | Mean MSE: {mean_mse:.4f}")
        
        # Check sparsity
        active_terms = np.sum(lasso_cv.coef_ != 0)
        total_terms = len(lasso_cv.coef_)
        print(f" -> Terms: {active_terms} active out of {total_terms} ({(total_terms-active_terms)/total_terms*100:.1f}% zeroed)")
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_degree = degree
            best_alpha = lasso_cv.alpha_
            
    print(f"\n=> Optimal Degree: {best_degree} with Lasso alpha={best_alpha:.4f}")
    
    # 1. Bias-Variance Plot
    plt.figure(figsize=(10, 5))
    plt.plot(degrees_to_test, cv_mses, marker='o', color='green', label='CV MSE')
    plt.yscale('log')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Validation MSE (Log Scale)')
    plt.title('var1: Bias-Variance Curve vs Degree')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'var1_bias_variance.png'), bbox_inches='tight')
    plt.close()

    print("\n--- Training Final Model ---")
    final_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=best_alpha, max_iter=20000))
    ])
    final_pipeline.fit(X, y)
    
    # 2. Residuals Plot
    y_pred_cv = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        pipeline_cv = Pipeline([
            ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=best_alpha, max_iter=20000))
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
    axes[0].set_title('var1: Actual vs Predicted')
    
    # Residuals Histogram
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_title('var1: Residual Distribution')
    
    plt.savefig(os.path.join(PLOT_DIR, 'var1_residuals.png'), bbox_inches='tight')
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
