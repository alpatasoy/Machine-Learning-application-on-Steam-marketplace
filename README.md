Salutes,

This is an attempt at developing an end-to-end solution to a real life problem example.
The dataset is retrieved from kaggle.com/datasets/fronkongames/steam-games-dataset
Here, I will try to further refine this study. You are more than welcome to provide any learning opportunities.
I also look forward to your insights.
Note:   I have removed personal opinions, insights and domain expertise as much as possible (basically any markdown fields in the Jupyter Notebook) to better focus on the methods i've utilized.
        I also would like to exchange opinions on the proposed solution outcomes and methods I've used in a discussion.

Special thanks to my mom (of course), and to my Data Science teachers.

# Steam End-to-End Regression Project

This project builds and evaluates multiple machine learning regression models to predict **`DEOM`** from a cleaned Steam dataset (`steam_cleaned.csv`). The workflow covers data loading, train/test splitting, baseline model comparison, cross-validation, hyperparameter tuning, final model selection, and feature-importance visualization.

## Project Overview

The notebook (`Steam-end2end.ipynb`) treats the problem as a **supervised regression task**:

- **Input:** engineered numeric and one-hot encoded Steam game features from `steam_cleaned.csv`
- **Target:** `DEOM`
- **Goal:** compare several regression approaches and select the best-performing model based on evaluation metrics

The notebook tests both linear baselines and tree/boosting-based ensemble models.

## Models Included

### Baseline models
- Linear Regression
- Ridge Regression
- Lasso Regression

### Ensemble / boosting models
- Random Forest Regressor
- Gradient Boosting Regressor
- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor

Each model is wrapped in a `scikit-learn` `Pipeline` with `MinMaxScaler`.

## Evaluation Metrics

The notebook evaluates models using:

- **R² Score**
- **Mean Squared Error (MSE)**
- **Mean Absolute Percentage Error (MAPE)**

It also uses **5-fold cross-validation** and **GridSearchCV** for hyperparameter tuning.

## Saved Results from the Notebook

### Test-set performance before tuning

| Model | R² | MSE |
|---|---:|---:|
| Linear Regression | 0.4534 | 3.9733 |
| Ridge Regression | 0.4528 | 3.9771 |
| Lasso Regression | 0.2393 | 5.5291 |
| Random Forest | 0.7039 | 2.1522 |
| Gradient Boosting | 0.7180 | 2.0497 |
| LightGBM | 0.7347 | 1.9280 |
| XGBoost | 0.7320 | 1.9483 |
| CatBoost | 0.7366 | 1.9147 |

### Cross-validation results for linear baselines

| Model | Mean CV R² | Std |
|---|---:|---:|
| Linear Regression | 0.4418 | 0.0135 |
| Ridge Regression | 0.4440 | 0.0095 |
| Lasso Regression | 0.2398 | 0.0022 |

### Best parameters found with GridSearchCV

- **Random Forest:** `max_depth=10`, `n_estimators=300`  
  Best CV R²: **0.7279**
- **Gradient Boosting:** `learning_rate=0.1`, `max_depth=5`, `n_estimators=200`  
  Best CV R²: **0.7338**
- **LightGBM:** `learning_rate=0.1`, `num_leaves=30`  
  Best CV R²: **0.7344**
- **XGBoost:** `learning_rate=0.1`, `max_depth=5`, `n_estimators=200`  
  Best CV R²: **0.7334**
- **CatBoost:** `depth=10`, `iterations=100`, `learning_rate=0.1`  
  Best CV R²: **0.7301**

### Final selected model

The notebook selects **LightGBM** as the final model:

- **R²:** `0.7351`
- **MSE:** `1.9254`

It then extracts feature importance values from the fitted LightGBM model and visualizes the top features.

## Repository Structure

```text
.
├── Steam-end2end.ipynb   # Main notebook with training, evaluation, tuning, and visualization
└── steam_cleaned.csv     # Input dataset used by the notebook
```


## Workflow Summary

1. Load the cleaned Steam dataset
2. Separate features and target (`DEOM`)
3. Split data into train and test sets
4. Train baseline and ensemble models
5. Compare models on R², MSE, and MAPE
6. Perform cross-validation
7. Tune selected ensemble models with grid search
8. Choose the best model
9. Plot feature importances

## Notes

- The notebook currently assumes `steam_cleaned.csv` already exists and is preprocessed.
- The reported **MAPE values are extremely large**, which often indicates that the target contains zeros or values very close to zero. In practice, this makes MAPE hard to interpret for this dataset, so **R²** and **MSE** are more reliable summary metrics here.
- Only the linear models are explicitly cross-validated in a separate section, while the ensemble models are tuned via `GridSearchCV`.
- The final model variable is set to `grid_lgbm.best_estimator_`.
