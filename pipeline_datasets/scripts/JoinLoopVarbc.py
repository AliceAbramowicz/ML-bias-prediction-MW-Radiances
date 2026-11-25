#!/usr/bin/env python3
"""
Join the VarBC dataset to the per-cycle merged ODB tables.

Reads  VARBC_OUTPUT_FILE (one row per predictor) & MERGED_ODB_DIR/merged_ODB_$DOMAIN_$YEAR.csv
Writes FINAL_DF 

Per (time, sat, sensor, channel) it:
- summarises statistics of your variables of interest from the odb for satellite data 
- merges the result onto VarBC params
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def require_env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable: {name}")
    return value

# Column layout of merged_ODB_*.csv in order
COLUMNS = ["date", "time", "lat", "lon", "sat", "sensor", "channel", "fg_depar", "an_depar", "biascorr_fg", "biascorr", "obs", "zenith", "azimuth"]
# alternative layout depending on what was selected in odb request: ML-bias-prediction-MW-Radiances/pipeline_datasets/scripts/WriteODBreq_sat.sh:
COLUMNS_96 = ["date", "time", "lat", "lon", "sat", "sensor", "channel",
                   "fg_depar", "an_depar", "biascorr_fg", "obs"]
COLUMNS_99 = ["date", "time", "lat", "lon", "sat", "sensor", "channel",
                   "fg_depar", "an_depar", "biascorr_fg", "biascorr", "obs",
                   "zenith", "azimuth"]

GROUP_KEYS = ["time", "sat", "sensor", "channel"]
STAT_COLS = ["fg_depar", "obs", "an_depar", "biascorr_fg"]
MERGE_KEYS = ["time", "sat", "sensor", "channel", "ndata", "npred", "pred_id"]

def build_agg_spec():
    # named aggregation
    spec = {}
    for col in STAT_COLS:
        spec[f"mean_{col}"] = (col, "mean")
        spec[f"var_{col}"] = (col, "var")
        spec[f"q01_{col}"] = (col, lambda s: s.quantile(0.01))
        spec[f"q05_{col}"] = (col, lambda s: s.quantile(0.05))
        # spec[f"q10_{col}"] = (col, lambda s: s.quantile(0.1))
        spec[f"q25_{col}"] = (col, lambda s: s.quantile(0.25))
        # spec[f"q30_{col}"] = (col, lambda s: s.quantile(0.3))
        # spec[f"q40_{col}"] = (col, lambda s: s.quantile(0.4))
        spec[f"q50_{col}"] = (col, "median")
        # spec[f"q60_{col}"] = (col, lambda s: s.quantile(0.6))
        spec[f"q75_{col}"] = (col, lambda s: s.quantile(0.75))
        # spec[f"q80_{col}"] = (col, lambda s: s.quantile(0.8))
        # spec[f"q90_{col}"] = (col, lambda s: s.quantile(0.9))
        spec[f"q95_{col}"] = (col, lambda s: s.quantile(0.95))
        spec[f"q99_{col}"] = (col, lambda s: s.quantile(0.01))
    return spec

AGG_SPEC = build_agg_spec()

def ccma_to_varbc_time(filename):
    """merged_ODB_$domain_$YYYYMMDDHH.csv --> 'YYYYMMDD_HHMMSS'"""
    stamp = Path(filename).stem.rsplit("_", 1)[-1]
    if len(stamp) != 10 or not stamp.isdigit():
        return None
    return f"{stamp[:8]}_{stamp[8:]}0000"

def main():
    VARBC_FILE = require_env("VARBC_OUTPUT_FILE")
    ODB_DIR = require_env("SAT_TABLES_DIR")
    OUTPUT_FILE = os.environ.get("FINAL_DF")

    varbc_df = pd.read_csv(VARBC_FILE, index_col=False)
    print(f"VARBC rows: {len(varbc_df)}")

    txt_files = sorted(f for f in os.listdir(ODB_DIR) if f.endswith(".txt"))
    print(f"Found {len(txt_files)} ODB file(s) in {ODB_DIR}")

    frames = []
    n_skipped = 0

    for filename in txt_files:
        file_path = os.path.join(ODB_DIR, filename)
        time = ccma_to_varbc_time(filename)
        if time is None:
            print(f"Skipping {filename}: cannot parse timestamp")
            n_skipped += 1
            continue

        try:
            ccma_df = pd.read_csv(file_path, sep=r"\s+", index_col=False)
        except Exception as exc:
            print(f"Error reading {file_path}: {exc}")
            n_skipped += 1
            continue

        n_cols = len(ccma_df.columns)
        if n_cols == len(COLUMNS):
            COLUMNS = COLUMNS
        elif n_cols == len(COLUMNS_96):
            COLUMNS = COLUMNS_96
        elif n_cols == len(COLUMNS_99):
            COLUMNS = COLUMNS_99
        else:
            print(f"Skipping {filename}: expected {len(COLUMNS_96)} or {len(COLUMNS_99)} columns, got {n_cols}")
            n_skipped += 1
            continue

        ccma_df.columns = COLUMNS
        ccma_df["time"] = time

        stats = ccma_df.groupby(GROUP_KEYS, as_index=False).agg(**AGG_SPEC)
        frames.append(stats.round(6))
        print(f"{filename}: {len(ccma_df)} rows --> {len(stats)} data groups")

    if not frames:
        sys.exit(f"No usable ODB files in {ODB_DIR}")

    all_dates = pd.concat(frames, ignore_index=True)

    # Collapse groups that appear in more than one file at overlapping times
    all_dates = all_dates.groupby(GROUP_KEYS, as_index=False).mean()
    print(f"Aggregated ODB rows: {len(all_dates)} "
          f"({n_skipped} file(s) skipped)")

    merged_df = pd.merge(varbc_df, all_dates, how="inner", on=GROUP_KEYS)
    if merged_df.empty:
        sys.exit("Merge produced no rows: check that VARBC and ODB time/sat/sensor/channel keys agree")
    merged_df = merged_df.groupby(MERGE_KEYS, as_index=False).mean()

    merged_df["time_formatted"] = pd.to_datetime(merged_df["time"], format="%Y%m%d_%H%M%S")
    # min_time = merged_df["time_formatted"].min()
    # max_time = merged_df["time_formatted"].max()
    # time_range = max_time - min_time

    days_in_year = np.where(merged_df["time_formatted"].dt.is_leap_year, 366.0, 365.0)
    day_fraction = (merged_df["time_formatted"].dt.dayofyear - 1)
    year_angle = 2 * np.pi * day_fraction / days_in_year

    merged_df["time_cos"] = np.cos(year_angle).round(6)
    merged_df["time_sin"] = np.sin(year_angle).round(6)

    merged_df["cycle"] = merged_df["time_formatted"].dt.hour
    merged_df = merged_df.drop(columns=["time_formatted"])
    merged_df = merged_df.drop_duplicates()

    tmp_path = OUTPUT_FILE + ".tmp"
    merged_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, OUTPUT_FILE)
    print(f"Wrote {len(merged_df)} rows x {len(merged_df.columns)} columns to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
