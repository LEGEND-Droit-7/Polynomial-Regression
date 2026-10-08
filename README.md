# Polynomial Regression: Industrial Optimization & Thermal Mapping

This repository contains the code and methodology for Machine Learning Assignment 1 (Polynomial Regression). The project solves two distinct real-world predictive modeling problems using regularized polynomial regression techniques.

## Project Overview

### Phase 1: Steam Turbine Net Power Output Optimization (`var1`)
- **Objective:** Predict the power output of a steam turbine based on 6 operational controls (e.g., valve adjustments, coolant flow).
- **Methodology:** Because the physical variables interact mechanically, the underlying relationship is highly sparse (i.e., a five-way multiplication of unrelated valves has no physical meaning). We applied **Lasso Regression (L1 Penalty)** combined with Polynomial Features. 
- **Result:** Lasso successfully acted as an automated feature selector, zeroing out 76.4% of the mathematical noise at Degree 5. The final model retained only the physically meaningful interactions, achieving an $R^2$ of **0.9659**.

### Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)
- **Objective:** Predict the thermal anomaly score across a survey block using 3 spatial coordinates ($x_1, x_2, x_3$).
- **Methodology:** Heat diffusion physics follows steady-state conduction, meaning the thermal map is continuous and dense in all directions. Dropping variables creates artificial "steps" in the field. Therefore, we used **Ridge Regression (L2 Penalty)** to smoothly shrink coefficients without discarding them.
- **Result:** The dense thermal map benefited from heavy mathematical interactions. The model steadily improved up to Degree 10, where Ridge perfectly smoothed the predictions to achieve an $R^2$ of **0.9941**.

## Key Highlights
- **Automated Hyperparameter Tuning:** Both phases utilized `LassoCV` and `RidgeCV` to sweep hundreds of potential penalty parameters ($\alpha$) natively within 5-fold cross-validation.
- **Dimensionality Control:** `StandardScaler` was rigidly applied to prevent polynomial coefficient explosion.

## Repository Structure
- `phase1_steam_turbine/`: Code and generated bias-variance/residual plots for the Lasso model.
- `phase2_thermal_mapping/`: Code and generated bias-variance/residual plots for the Ridge model.
- `BT2024203_pred_var1.csv` & `BT2024203_pred_var2.csv`: Final test predictions.
- `BT2024203_Report.md`: Full assignment evaluation and conclusion report.
