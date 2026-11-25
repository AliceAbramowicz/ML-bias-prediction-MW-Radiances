import os
import pickle
import random
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(22)
random.seed(22)

# CPU threads: match the SLURM allocation (falls back to visible cores)
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0)))) # 256 threads default
print(f"Using {N_JOBS} threads")

# Load data
# -----------------------------
X_train = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/X_train_stats_scaled.csv")
X_val = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/X_val_stats_scaled.csv")
X_test = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/X_test_stats_scaled.csv")
y_train = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/y_train_stats_scaled.csv").values.ravel()
y_val = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/y_val_stats_scaled.csv").values.ravel()
y_test = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/y_test_stats_scaled.csv").values.ravel()

print(X_train.columns)
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

suffix = ""

SEARCH_PKL = f"RF_search{suffix}.pkl"     # hyperparameter search outcome
RESULTS_PKL = f"RF_results{suffix}.pkl"   # final model evaluation
MODEL_JSON = f"RF_model{suffix}.json"
MODEL_PKL = f"RF_model{suffix}.pkl"

param_grid = list(ParameterGrid({
    "max_features": [0.4, 0.5, 0.6],
    "min_samples_leaf": [2, 5, 15, 30],
}))

# Search settings: fewer trees and a subsampled bootstrap to keep it cheap
SEARCH_KW = dict(
    n_estimators=100,
    max_samples=0.5,
    bootstrap=True,
    n_jobs=N_JOBS,
    random_state=22,
)

if os.path.exists(SEARCH_PKL):
    print(f"Loading existing grid search from {SEARCH_PKL}")
    with open(SEARCH_PKL, "rb") as f:
        search_results = pickle.load(f)
    best_params = search_results["best_params"]
    print(search_results["grid_df"])
else:
    print(f"Running grid search over {len(param_grid)} configurations...")
    rows = []
    for i, params in enumerate(param_grid, 1):
        rf = RandomForestRegressor(**SEARCH_KW, **params)
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        depths = [t.get_depth() for t in rf.estimators_]
        rows.append({**params, "val_rmse": val_rmse, "val_r2": val_r2, "mean_depth": np.mean(depths)})
        print(f"[{i}/{len(param_grid)}] {params} \nval_rmse={val_rmse:.4f}  \nval_r2={val_r2:.4f} \nmean_depth={np.mean(depths):.1f}")
        del rf

    grid_df = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    best_params = {k: grid_df.loc[0, k] for k in ["max_features", "min_samples_leaf"]}
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    search_results = {"best_params": best_params,
                      "best_val_rmse": float(grid_df.loc[0, "val_rmse"]),
                      "grid_df": grid_df}
    with open(SEARCH_PKL, "wb") as f:
        pickle.dump(search_results, f)
    print("\nGrid results:\n", grid_df)
print("Best hyperparameters found:", best_params)

# -----------------------------
# Final model: more trees, full bootstrap
# -----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_samples=None,
    bootstrap=True,
    n_jobs=N_JOBS,
    random_state=22,
    **best_params,
)

model.fit(X_train, y_train)

val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
print(f"Final model val RMSE: {val_rmse:.5f}")
 
with open(MODEL_PKL, "wb") as f:
    pickle.dump(model, f)
print(f"Saved model to {MODEL_PKL}")

# -----------------------------
# Evaluate on test set
# -----------------------------
y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)
test_mse = mean_squared_error(y_test, y_pred)
val_mse = mean_squared_error(y_val, y_pred_val)
test_rmse = np.sqrt(test_mse)
print(f"Val MSE scaled; {val_mse:.6f}")
print(f"Test MSE scaled: {test_mse:.6f}")
test_r2 = r2_score(y_test, y_pred)
val_r2 = r2_score(y_val, y_pred_val)
print(f"Val R2 scaled: {val_r2:.6f}")
print(f"Test R2 scaled: {test_r2:.6f}")

baseline_mse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))
print(f"Baseline (train mean) MSE: {baseline_mse:.6f}")
print(f"Prediction spread: std={y_pred.std():.6f} (vs y_test std={y_test.std():.6f})")

result_dict = {
    "best_estimator": MODEL_PKL,
    "best_params": best_params,
    "y_test_scaled": y_test,
    "y_pred_scaled": y_pred,
    "scaled_metrics": {"mse": test_mse, "val_mse": val_mse, "r2": test_r2, "val_r2": val_r2, "baseline_mse": baseline_mse},
    "feature_importances": pd.Series(model.feature_importances_, index=X_train.columns),
}

with open(RESULTS_PKL, "wb") as f:
    pickle.dump(result_dict, f)
 
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)
print(f"Results saved to {RESULTS_PKL}")
