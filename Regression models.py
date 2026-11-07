
# ============================================================
# CAR SALES REGRESSION: kNN AND DECISION TREE
# ============================================================

# 1. IMPORT LIBRARIES

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD THE ALREADY-CLEANED DATASET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "auto_prices_cleaned.csv"

VIZ_DIR = PROJECT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Convert date column
df["month_year"] = pd.to_datetime(
    df["month_year"],
    format="%Y-%m-%d"
)

# Order data chronologically
df = df.sort_values("month_year").reset_index(drop=True)


# ============================================================
# 3. DEFINE PREDICTORS AND TARGET
# ============================================================

features = [
    "new_price",
    "used_price",
    "inflation_rate",
    "interest_rate"
]

target = "units_sold"


# ============================================================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# Train on 2019–2022
train = df[
    (df["month_year"] >= "2019-01-01") &
    (df["month_year"] < "2023-01-01")
].copy()

# Test on 2023
test = df[
    (df["month_year"] >= "2023-01-01") &
    (df["month_year"] < "2024-01-01")
].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\n===== TRAIN / TEST SPLIT =====")
print("Training observations:", len(train))
print("Testing observations:", len(test))


# ============================================================
# 5. TIME-SERIES CROSS-VALIDATION
# ============================================================

# Expanding training windows, preserving chronological order
tscv = TimeSeriesSplit(n_splits=5)

print("\n===== TIME-SERIES CV FOLDS =====")

for fold, (train_idx, val_idx) in enumerate(
    tscv.split(X_train), start=1
):
    print(
        f"Fold {fold}: "
        f"{len(train_idx)} training observations, "
        f"{len(val_idx)} validation observations"
    )


# ============================================================
# 6. kNN REGRESSION: AUTOMATIC BEST k
# ============================================================

# Pipeline scales the predictors before kNN
knn_pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)

# With 48 training observations and 5 CV splits,
# the first training fold contains 8 observations.
# Therefore test k = 1 to 8.
param_grid = {
    "kneighborsregressor__n_neighbors": range(1, 9)
}

grid_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Find best k using only training data
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_[
    "kneighborsregressor__n_neighbors"
]
print("\n===== kNN GRID SEARCH RESULTS =====")
print("Best number of neighbors (k):", best_k)

# Best fitted pipeline (scaler + kNN)
knn_model = grid_search.best_estimator_

# Predict 2023 sales
knn_predictions = knn_model.predict(X_test)

# Calculate evaluation metrics
knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_rmse = np.sqrt(
    mean_squared_error(y_test, knn_predictions)
)
knn_r2 = r2_score(y_test, knn_predictions)

print("\n===== kNN TEST RESULTS =====")
print(f"MAE:  {knn_mae:.2f}")
print(f"RMSE: {knn_rmse:.2f}")
print(f"R²:   {knn_r2:.4f}")



# ============================
# 8. DECISION TREE REGRESSION 
# ============================

# Create Decision Tree model
tree_pipeline = DecisionTreeRegressor(
    min_samples_leaf=3,
    random_state=42
)

# Define maximum tree depths to test
tree_param_grid = {
    "max_depth": [2, 3, 4, 5, None]
}

# GridSearchCV using chronological cross-validation
tree_grid_search = GridSearchCV(
    estimator=tree_pipeline,
    param_grid=tree_param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Find best depth using only 2019–2022 training data
tree_grid_search.fit(X_train, y_train)

# Get best maximum depth
best_depth = tree_grid_search.best_params_["max_depth"]

# Get optimized Decision Tree model
# GridSearchCV automatically refits on all training data
tree_model = tree_grid_search.best_estimator_

# Get average cross-validation MAE
tree_cv_mae = -tree_grid_search.best_score_

print("\n===== DECISION TREE GRID SEARCH RESULTS =====")
print("Best maximum depth:", best_depth)
print(f"Best CV MAE: {tree_cv_mae:.2f}")

# Predict 2023 car sales
tree_predictions = tree_model.predict(X_test)

# Calculate evaluation metrics
tree_mae = mean_absolute_error(y_test, tree_predictions)

tree_rmse = np.sqrt(
    mean_squared_error(y_test, tree_predictions)
)

tree_r2 = r2_score(y_test, tree_predictions)

print("\n===== DECISION TREE TEST RESULTS (2023) =====")
print(f"MAE:  {tree_mae:.2f}")
print(f"RMSE: {tree_rmse:.2f}")
print(f"R²:   {tree_r2:.4f}")



# ============================================================
# 9. COMPARE BOTH REGRESSION MODELS
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "kNN Regression",
        "Decision Tree Regression"
    ],
    
    "Test MAE": [
        knn_mae,
        tree_mae
    ],
    "Test RMSE": [
        knn_rmse,
        tree_rmse
    ],
    "Test R²": [
        knn_r2,
        tree_r2
    ]
})

print("\n===== MODEL COMPARISON =====")
print(comparison.round(3).to_string(index=False))


# ============================================================
# 10. ACTUAL VS PREDICTED SALES (BOTH MODELS)
# ============================================================

results = pd.DataFrame({
    "month_year": test["month_year"].values,
    "actual": y_test.values,
    "knn_predicted": knn_predictions,
    "tree_predicted": tree_predictions
})

plt.figure(figsize=(12, 6))

plt.plot(
    results["month_year"],
    results["actual"],
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

plt.plot(
    results["month_year"],
    results["knn_predicted"],
    marker="s",
    linestyle="--",
    label="kNN Predictions"
)

plt.plot(
    results["month_year"],
    results["tree_predicted"],
    marker="^",
    linestyle="--",
    label="Decision Tree Predictions"
)

plt.title("Actual vs Predicted Car Sales (2023)")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    VIZ_DIR / "actual_vs_predicted_2023.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 14. FINAL SUMMARY
# ============================================================

##print("\n===== FINAL RESULTS =====")
print(comparison.round(3).to_string(index=False))


# ============================================================
# SAVE REGRESSION RESULTS TO README.md
# ============================================================

README_PATH = PROJECT_DIR / "README.md"

# Create Markdown table
table_rows = [
    "| Model | Test MAE | Test RMSE | Test R² |"
]

for _, row in comparison.iterrows():
    table_rows.append(
        f"| {row['Model']} "
        f"| {row['Test MAE']:.2f} "
        f"| {row['Test RMSE']:.2f} "
        f"| {row['Test R²']:.4f} |"
    )

results_table = "\n".join(table_rows)

# Create results section
results_section = f"""
## Regression Model Results

{results_table}
"""
# Append results to README
with open(README_PATH, "a", encoding="utf-8") as file:
    file.write("\n" + results_section)

print("\nRegression results saved to README.md")