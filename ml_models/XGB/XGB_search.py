import os
import pickle
import random
 
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
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

suffix = ""

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

print(f"y_train  mean={y_train.mean():.5f}  std={y_train.std():.5f}"
      f"min={y_train.min():.4f}  max={y_train.max():.4f}"
      f"median|y|={np.median(np.abs(y_train)):.5f}")

SEARCH_PKL = f"XGB_search{suffix}.pkl"     # hyperparameter search outcome
RESULTS_PKL = f"XGB_results{suffix}.pkl"   # final model evaluation
MODEL_JSON = f"XGB_model{suffix}.json"

N_TRIALS = 60
 
FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "random_state": 22,
    "n_jobs": N_JOBS,
    "learning_rate": 0.01,
    "gamma": 0.0,
    "reg_alpha": 0.0,
}

# Bayesian hyperparameter search
# -----------------------------
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 6, 12, step=1), # raise it to go past categories
        "min_child_weight": trial.suggest_float("min_child_weight", 2, 10, step=1),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8, step=0.1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 0.8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5, step=0.1), # L2 regularization
        "reg_alpha": trial.suggest_float("reg_lambda", 0, 1, step=0.1),
    }

    model = XGBRegressor(
        n_estimators=200, # upper bound
        early_stopping_rounds=10,
        **FIXED_PARAMS,
        **params,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    trial.set_user_attr("best_iteration", model.best_iteration)
    return model.best_score  # validation MSE at the best iteration

if os.path.exists(SEARCH_PKL):
    print(f"Loading existing hyperparameter search from {SEARCH_PKL}")
    with open(SEARCH_PKL, "rb") as f:
        search_results = pickle.load(f)
    best_params = search_results["best_estimator"]

else:
    print("No existing search found. Running hyperparameter search...")
    sampler = optuna.samplers.TPESampler(seed=22)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    search_results = {
        "best_params": best_params,
        "best_val_mse": study.best_value,
        "best_iteration": study.best_trial.user_attrs["best_iteration"],
        "trials_df": study.trials_dataframe(),
    }
    with open(SEARCH_PKL, "wb") as f:
        pickle.dump(search_results, f)
    print("Best validation MSE:", study.best_value)
 
print("Best hyperparameters found:", best_params)

# Final training with early stopping
# -----------------------------
model = XGBRegressor(
    n_estimators=3000,                # upper bound
    early_stopping_rounds=45,        # more patience than during the search
    **FIXED_PARAMS,
    **best_params,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)
print("Best iteration:", model.best_iteration)
print("Best validation MSE:", model.best_score)
model.save_model(MODEL_JSON)

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

# A model with too much regularization predicts a constant
# Compare against the train-mean baseline to catch that.
baseline_mse = mean_squared_error(y_test, np.full_like(np.asarray(y_test, dtype=float).ravel(), y_train.mean()))
print(f"Baseline (train mean) MSE: {baseline_mse:.6f}")
print(f"Prediction spread: std={y_pred.std():.6f} (near 0 means a degenerate model)")

# Save model and results
# -----------------------------
result_dict = {
    "best_estimator": MODEL_JSON,
    "best_params": {**FIXED_PARAMS, **best_params},
    "best_iteration": model.best_iteration,
    "scaled_metrics": {"mse": test_mse, "r2": test_r2, "val_r2": val_r2, "val_mse": val_mse, "baseline_mse": baseline_mse},
    "y_test_scaled": np.asarray(y_test).flatten(),
    "y_pred_scaled": y_pred.flatten(),
    "evals_result": model.evals_result(),
    "feature_importances": pd.Series(model.feature_importances_, index=X_train.columns),
}

with open(RESULTS_PKL, "wb") as f:
    pickle.dump(result_dict, f)

print("Best hyperparameters:", best_params)
print("Test RMSE:", test_rmse)
print("Test MSE:", test_mse)
print("Test R2:", test_r2)
print(f"Results saved to XGB_results{suffix}.pkl")
