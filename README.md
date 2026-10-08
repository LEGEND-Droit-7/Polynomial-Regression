# Polynomial Regression: Industrial Optimization & Thermal Mapping

Welcome to the repository for **Machine Learning Assignment 1: Polynomial Regression**. This project tackles two distinct real-world predictive modeling problems using regularized polynomial regression techniques, emphasizing the intersection between mathematical models and physical intuition.

## 📖 Project Overview

### Phase 1: Steam Turbine Net Power Output Optimization (`var1`)
- **Objective:** Predict the net power output of an industrial steam turbine utilizing 6 operational controls (e.g., high-pressure valve adjustments, coolant flow, re-injection pressure).
- **Physical Intuition:** Industrial machinery relies on specific mechanical interactions. A highly complex 5-way mathematical product of completely unrelated valves generally carries zero physical meaning. Because of this, the underlying mathematical relationship is expected to be **sparse**.
- **Methodology:** We applied **Lasso Regression (L1 Penalty)** coupled with Polynomial Features. 
- **Results:** Lasso successfully acted as a strict, automated feature selector. When testing a Degree 5 polynomial (which generates 461 potential interaction terms), Lasso correctly zeroed out 76.4% of the mathematical noise. By retaining only the 109 physically meaningful interactions, the model achieved a highly robust **CV $R^2$ of 0.9659** without succumbing to the curse of dimensionality.

### Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)
- **Objective:** Predict the continuous thermal anomaly score across a survey block using 3 specific spatial coordinates ($x_1, x_2, x_3$).
- **Physical Intuition:** Heat diffusion physics follows steady-state conduction. This dictates that thermal maps must be continuous and smooth in all directions. If a model completely drops specific spatial variables, it creates artificial, impossible "steps" in the physical temperature field. Thus, the relationship is expected to be **dense**.
- **Methodology:** We utilized **Ridge Regression (L2 Penalty)** to softly shrink coefficients without discarding them entirely, preserving the continuous field.
- **Results:** Benefiting from a smaller input feature space and the dense nature of Ridge, we were able to safely push the model to heavy mathematical interactions. The model steadily improved up to **Degree 10**, where the Ridge penalty perfectly smoothed the predictions to achieve a near-flawless **CV $R^2$ of 0.9941**.

## 🚀 Key Technical Highlights
- **Automated Hyperparameter Tuning:** Both phases utilized `LassoCV` and `RidgeCV` to sweep across hundreds of potential penalty parameters ($\alpha$). This occurred natively within a rigorous **5-fold cross-validation** framework to prevent any data leakage.
- **Dimensionality Control:** `StandardScaler` was rigidly applied to all base features before polynomial generation to completely prevent polynomial coefficient explosion and numerical instability.

## 📁 Repository Structure

The repository has been structured cleanly to separate raw data, model scripts, evaluation artifacts, and final predictions:

```text
├── BT2024203_ML_Dataset/
│   ├── BT2024203_train_var1.csv      # Phase 1 training data
│   ├── BT2024203_test_var1.csv       # Phase 1 testing data
│   ├── BT2024203_train_var2.csv      # Phase 2 training data
│   └── BT2024203_test_var2.csv       # Phase 2 testing data
│
├── phase1_steam_turbine/
│   ├── train_var1.py                 # Pipeline and LassoCV training script
│   ├── var1_bias_variance.png        # Evaluated MSE plot across degrees
│   └── var1_residuals.png            # Actual vs Predicted error spread
│
├── phase2_thermal_mapping/
│   ├── train_var2.py                 # Pipeline and RidgeCV training script
│   ├── var2_bias_variance.png        # Evaluated MSE plot across degrees
│   └── var2_residuals.png            # Actual vs Predicted error spread
│
├── predictions/
│   ├── BT2024203_pred_var1.csv       # Final inference results for Phase 1
│   └── BT2024203_pred_var2.csv       # Final inference results for Phase 2
│
└── README.md                         # Project documentation 
```
