import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from xgboost import XGBRegressor, DMatrix, train, callback

np.random.seed(22)
random.seed(22)

# Load data
# -----------------------------
X = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
y = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv").values.ravel()
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv").values.ravel()

# Train / Validation Split
# -----------------------------
val_fraction = 0.1
split_idx = int(len(X) * (1 - val_fraction))

X_train_full = X.iloc[:split_idx]
y_train_full = y[:split_idx]
X_val = X.iloc[split_idx:]
y_val = y[split_idx:]

# Hyperparameter grid (manual search to save memory)
# -----------------------------
param_grid = [
    {
        "max_depth": d,
        "learning_rate": lr,
        "gamma": g,
        "subsample": ss,
        "colsample_bytree": cs,
        "reg_alpha": ra
    }
    for d in [3, 4, 5]
    for lr in [0.05, 0.1, 0.15, 0.2]
    for g in [0, 0.05, 0.1]
    for ss in [0.5, 0.65, 0.8]
    for cs in [0.5, 0.65]
    for ra in [0.1, 0.15, 0.2]
]


# TimeSeriesSplit for CV (memory-safe)
# -----------------------------
tscv = TimeSeriesSplit(n_splits=3)

def evaluate_params(params):
    fold_scores = []
    for train_idx, val_idx in tscv.split(X_train_full):
        X_tr, X_cv = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_cv = y_train_full[train_idx], y_train_full[val_idx]

        model = XGBRegressor(
            n_estimators=350,  
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            random_state=22,
            **params
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_cv)
        fold_scores.append(mean_squared_error(y_cv, preds))

        # Clean memory
        del model, X_tr, X_cv, y_tr, y_cv, preds

    return np.mean(fold_scores)

# Manual hyperparameter search
# -----------------------------
SEARCH_PKL = "XGB_results.pkl"
if os.path.exists(SEARCH_PKL):
    print(f"Loading existing hyperparameter search from {SEARCH_PKL}")

    with open(SEARCH_PKL, "rb") as f:
        search_results = pickle.load(f)

    best_params = search_results["best_estimator"]

else:
    print("No existing search found. Running hyperparameter search...")
    best_score = float("inf")
    best_params = None
    for params in param_grid:
        score = evaluate_params(params)
        print("Params:", params, "CV MSE:", score)
        if score < best_score:
            best_score = score
            best_params = params

    print("Best hyperparameters found:", best_params)

# Final training on full training data with early stopping
# -----------------------------
dtrain = DMatrix(X_train_full, label=y_train_full)
dval = DMatrix(X_val, label=y_val)

final_params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    **best_params
}

early_stop = callback.EarlyStopping(rounds=25, save_best=True, maximize=False)

bst = train(
    params=final_params,
    dtrain=dtrain,
    num_boost_round=5000, # upper bound
    evals=[(dtrain, "train"), (dval, "val")],
    callbacks=[early_stop]
)

# Evaluate on test set
# -----------------------------
dtest = DMatrix(X_test, label=y_test)
y_pred = bst.predict(dtest)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Save model and results
# -----------------------------
result_dict = {
    "best_estimator": final_params,
    "mse": test_mse,
    "r2": test_r2,
    "y_pred": y_pred
}

with open(SEARCH_PKL, "wb") as f:
    pickle.dump(result_dict, f)

print("Best hyperparameters:", best_params)
print("Test MSE:", test_mse)
print("Test R2:", test_r2)
print("Results saved to XGB_results.pkl")
