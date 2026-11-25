#!/usr/bin/env python3
"""
Build the train/test matrices from two csv datasets.

Run manually (not part of the scripts pipeline, since it requires two domains at once)

    python3 scripts/create_X_train_test.py
    python3 scripts/create_X_train_test.py --na-mode no-predxcov
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = "/perm/nld3863/ML-bias-prediction-MW-Radiances"
DEFAULT_OUTPUT_DIR = str(f"{REPO_ROOT}/datasets/final_df")
EXP = "nwp_cy43_exp"

CATEGORIES = ["pred_id", "sat", "sensor", "channel"]
DROP_FROM_FEATURES = ["param", "param0", "cycle", "time_formatted", "time"] # , "time"

def apply_na_mode(df_train, df_test, mode):
    """Four mutually exclusive ways of handling VarBC predictor covariances."""
    if mode == "five-pred":
        # Keep only 5-predictor rows; predxcov_6 is then always empty.
        df_train = df_train[df_train["npred"] == 5].drop(columns=["predxcov_6"])
        df_test = df_test[df_test["npred"] == 5].drop(columns=["predxcov_6"])
    elif mode == "drop-cols-with-missing":
        drop = [col for col in df_train.columns if col.startswith("predxcov_") and df[col].isna().any()]
        df_train = df_train.drop(columns=drop)
        df_test = df_test.drop(columns=drop)
    elif mode == "zero":
        for df in (df_train, df_test):
            cov = df.columns[df.columns.str.startswith("predxcov_")]
            df[cov] = df[cov].fillna(0)
    elif mode == "no-predxcov":
        drop = [c for c in df_train.columns if c.startswith("predxcov")]
        df_train = df_train.drop(columns=drop)
        df_test = df_test.drop(columns=drop)
    else:
        raise ValueError(f"unknown --na-mode: {mode}")
    return df_train, df_test

####### Check timestamps #############

def find_missing_timestamps(df, dataset_name):
    """Find, print, and calculate the proportion of missing timestamps in a df with 3-hourly data."""
    # Convert the 'time' column to datetime
    df['time_dt'] = pd.to_datetime(df['time'], format='%Y%m%d_%H%M%S')
    min_time = df['time_dt'].min()
    max_time = df['time_dt'].max()
    expected_times = pd.date_range(start=min_time, end=max_time, freq='3H')
    total_expected = len(expected_times)
    actual_times = df['time_dt'].sort_values().unique()
    total_actual = len(actual_times)
    missing_times = expected_times.difference(actual_times)
    total_missing = len(missing_times)
    proportion_missing = total_missing / total_expected if total_expected > 0 else 0

    if total_missing == 0:
        print(f"No missing timestamps in {dataset_name}.")
    else:
        print(f"Missing timestamps in {dataset_name} ({total_missing}/{total_expected} = {proportion_missing:.2%}):")
        for time in missing_times:
            print(time.strftime('%Y%m%d_%H%M%S'))

    df.drop(columns=['time_dt'], inplace=True)

def split_by_time(df, val_frac):
    """Chronological split on whole timestamps.
    Splitting on a row index cuts through whichever cycle straddles the
    boundary, putting the same timestamp on both sides. 
    """
    times = np.sort(df["time"].unique())
    n_val = int(round(len(times) * val_frac))
    print(f"{len(times)} timestamps in total, {n_val} in val")
    n_val = min(n_val, len(times))
    print(f"n_val still {n_val}")
    boundary = times[len(times) - n_val]
    print('time boundary:', boundary)
    train = df[df["time"] < boundary].copy()
    val = df[df["time"] >= boundary].copy()
    print(f"chronological split at {boundary}: "
          f"{len(times) - n_val} train cycles ({len(train)} rows), "
          f"{n_val} val cycles ({len(val)} rows)")
    print('val set:')
    print(val["time"])
    assert not (set(train["time"]) & set(val["time"])), \
        "a timestamp ended up in both train and validation"
    return train, val

def main():
    parser = argparse.ArgumentParser(description="Build train/test sets")
    parser.add_argument("--train", default=f"{DEFAULT_OUTPUT_DIR}/big_df_stats_dini.csv")
    parser.add_argument("--test", default=f"{DEFAULT_OUTPUT_DIR}/big_df_stats_{EXP}.csv")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--na-mode", default="no-predxcov",
                        choices=["zero", "drop-cols-with-missing", "five-pred", "no-predxcov"],
                        help="how to handle variable predictor covariances")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="fraction of training timestamps held out for validation")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_train = pd.read_csv(args.train).replace("", np.nan)
    df_test = pd.read_csv(args.test).replace("", np.nan)
    print(f"train {df_train.shape}, test {df_test.shape}")

    df_train, df_test = apply_na_mode(df_train, df_test, args.na_mode)

    print(f"Train shape: {df_train.shape}, test shape: {df_test.shape}")

########### Check missing data #################

    # change pandas display settings to show all rows/columns
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)

    # Find rows with any missing values & print them
    rows_with_nan_train = df_train[df_train.isna().any(axis=1)]
    rows_with_nan_test = df_test[df_test.isna().any(axis=1)]

    if not rows_with_nan_train.empty:
        print("Rows with missing values:")
        print(rows_with_nan_train)
        df_train = df_train.dropna()
        print(f"\nWARNING: Dropped {len(rows_with_nan_train)} rows from df_train due to missing values.")
    if not rows_with_nan_test.empty:
        print("Rows with missing values:")
        print(rows_with_nan_test)
        df_test = df_test.dropna()
        print(f"\nWARNING: Dropped {len(rows_with_nan_test)} rows from df_test due to missing values.")

    print(f"After removing NaNs: Train shape: {df_train.shape}, test shape: {df_test.shape}")

########### Check missing times ###########

    print("########## \n Checking for missing timestamps in df_train: \n ##########")
    find_missing_timestamps(df_train, "df_train")

    print("########## \nChecking for missing timestamps in df_test: \n ##########")
    find_missing_timestamps(df_test, "df_test")

########### add features ###########

    def add_cycle_features(df):
        angle = df["cycle"] * 2 * np.pi / 24
        block = pd.DataFrame({"cycle_cos": np.cos(angle).round(6),
                              "cycle_sin": np.sin(angle).round(6)},
                              index=df.index)
        return pd.concat([df, block], axis=1)

    df_train = add_cycle_features(df_train)
    df_test = add_cycle_features(df_test)

########### split train into val and train ###########
 
    df_train = df_train.sort_values("time")
    df_train_split, df_val = split_by_time(df_train, args.val_frac)
 
########### targets, scaled on the train part only ###########
 
    y_train = df_train_split["param"]
    y_val = df_val["param"]
    y_test = df_test["param"]
 
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
 
    X_train = df_train_split.drop(columns=DROP_FROM_FEATURES, errors="ignore")
    X_val = df_val.drop(columns=DROP_FROM_FEATURES, errors="ignore")
    X_test = df_test.drop(columns=DROP_FROM_FEATURES, errors="ignore")
 
########## Ensure same columns everywhere ##########
 
    numeric_columns = [c for c in X_train.columns if c not in CATEGORIES] # and c != "time"
 
    X_train = pd.get_dummies(X_train, columns=CATEGORIES)
    X_val = pd.get_dummies(X_val, columns=CATEGORIES)
    X_test = pd.get_dummies(X_test, columns=CATEGORIES)
 
    # Union of the three column sets, so every category is always present and nothing is dropped from any of them
    all_columns = list(dict.fromkeys(
        list(X_train.columns) + list(X_val.columns) + list(X_test.columns)))
    added_to_train = [c for c in all_columns if c not in X_train.columns]
    added_to_val = [c for c in all_columns if c not in X_val.columns]
    added_to_test = [c for c in all_columns if c not in X_test.columns]
    if added_to_train:
        print(f"added to train as 0: {added_to_train}")
    if added_to_val:
        print(f"added to val as 0: {added_to_val}")
    if added_to_test:
        print(f"added to test as 0: {added_to_test}")
 
    X_train = X_train.reindex(columns=all_columns, fill_value=0)
    X_val = X_val.reindex(columns=all_columns, fill_value=0)
    X_test = X_test.reindex(columns=all_columns, fill_value=0)
 
########## Normalize one-hot vectors to int ##########
 
    # avoid mix of True/False and 0 in the csv
    dummy_cols = [c for c in X_train.columns if c not in numeric_columns] # and c != "time"
    X_train[dummy_cols] = X_train[dummy_cols].astype("int8")
    X_val[dummy_cols] = X_val[dummy_cols].astype("int8")
    X_test[dummy_cols] = X_test[dummy_cols].astype("int8")
 
########## last sanity checks and savings datasets ##########

    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns), "train, val and test feature columns differ"
    assert not X_train.isna().any().any(), "NaN remains in X_train"
    assert not X_val.isna().any().any(), "NaN remains in X_val"
    assert not X_test.isna().any().any(), "NaN remains in X_test"
    assert len(X_train) == len(y_train) and len(X_val) == len(y_val) and len(X_test) == len(y_test), "X and y row counts disagree"

    # # X_train.to_csv(f"{args.outdir}/X_train_stats.csv", index=False)
    # # X_val.to_csv(f"{args.outdir}/X_val_stats.csv", index=False)
    # # X_test.to_csv(f"{args.outdir}/X_test_stats.csv", index=False)
    y_train.to_csv(f"{args.outdir}/y_train_stats.csv", index=False)
    y_val.to_csv(f"{args.outdir}/y_val_stats.csv", index=False)
    y_test.to_csv(f"{args.outdir}/y_test_stats.csv", index=False)
 
    X_train_scaled = X_train.copy().astype(float)
    X_val_scaled = X_val.copy().astype(float)
    X_test_scaled = X_test.copy().astype(float)
 
    scaler_X = StandardScaler()
    X_train_scaled[numeric_columns] = scaler_X.fit_transform(X_train_scaled[numeric_columns])
    X_val_scaled[numeric_columns] = scaler_X.transform(X_val_scaled[numeric_columns])
    X_test_scaled[numeric_columns] = scaler_X.transform(X_test_scaled[numeric_columns])

    joblib.dump(scaler_y, f"{args.outdir}/scaler_y.joblib")
    joblib.dump(scaler_X, f"{args.outdir}/scaler_X.joblib")
    with open(f"{args.outdir}/scaler_y.json", "w") as fh:
        json.dump({"mean": float(scaler_y.mean_[0]), 
                   "scale": float(scaler_y.scale_[0])}, fh, indent=1)
 
    print(f"After adding categories, cycles sin & cos, and dropping {DROP_FROM_FEATURES}:"
          f"\nTrain shape: {X_train.shape}, " \
          f"\nVal shape: {X_val.shape}, " \
          f"\ntesting shape: {X_test.shape}")
 
    X_train_scaled.to_csv(f"{args.outdir}/X_train_stats_scaled.csv", index=False)
    X_val_scaled.to_csv(f"{args.outdir}/X_val_stats_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{args.outdir}/X_test_stats_scaled.csv", index=False)
    pd.Series(y_train_scaled.flatten(), name="param").to_csv(
        f"{args.outdir}/y_train_stats_scaled.csv", index=False)
    pd.Series(y_val_scaled.flatten(), name="param").to_csv(
        f"{args.outdir}/y_val_stats_scaled.csv", index=False)
    pd.Series(y_test_scaled.flatten(), name="param").to_csv(
        f"{args.outdir}/y_test_stats_scaled.csv", index=False)
 
 
if __name__ == "__main__":
    main()
