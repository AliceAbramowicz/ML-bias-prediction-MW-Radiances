#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os, sys, json

# -----------------------------
# Environment setup
# -----------------------------
try:
    HM_DIR = os.environ["HM_DIR"]
    DOMAIN = os.environ["DOMAIN"]
    YEAR = os.environ["YEAR"]
    DAYS_JSON = json.loads(os.environ.get("DAYS_JSON", "{}"))
    MONTHS = os.environ.get("MONTHS", "").split()
    CYCLES = os.environ.get("CYCLES", "").split()
except KeyError as e:
    sys.exit(f"Missing environment variable: {e}")

SAT_DIR = os.environ.get("SAT_DIR", f"{HM_DIR}/datasets/sat_{YEAR}_tables")
ANCHOR_DIR = os.environ.get("ANCHOR_DIR", f"{HM_DIR}/datasets/anchors_{YEAR}_tables")
MERGED_ODB_DIR = os.environ.get("MERGED_ODB_DIR", f"{HM_DIR}/datasets/merged_{YEAR}_tables")

os.makedirs(MERGED_ODB_DIR, exist_ok=True)

# -----------------------------

# Configurable constants
# -----------------------------
THRESHOLDS_HPA = [90, 100, 180, 250, 270, 300, 350, 400, 500, 550, 600, 650, 800]
layer_halfwidth = 10  # default ±10 hPa except for top layers 3-9 (AMSUA channel 9 at 90 hPa) and 73-4(MWHS2 channel 4) otherwise they are empty
SENSORS = os.environ.get("SENSORS", "3,15,73").split(",")

for mm in MONTHS:
    days = [f"{int(d):02d}" for d in DAYS_JSON.get(str(mm).zfill(2), [])]
    for dd in days:
        for cy in CYCLES:
            sat_path = f"{SAT_DIR}/sat_{YEAR}{mm}{dd}{cy}_with_hPa.csv"
            anchor_path = f"{ANCHOR_DIR}/anchors_{YEAR}{mm}{dd}{cy}.txt"
            out_path = f"{MERGED_ODB_DIR}/merged_ODB_{DOMAIN}_{YEAR}{mm}{dd}{cy}.csv"
            # Check existence and skip if missing
            if not os.path.exists(anchor_path) or not os.path.exists(sat_path):
                print(f"Missing file, skipping: {YEAR}{mm}{dd}{cy}")
                continue
            if os.path.getsize(anchor_path) < 100 or os.path.getsize(sat_path) < 100:
                print(f"Empty file, skipping: {YEAR}{mm}{dd}{cy}")
                continue

            print(f"Processing: {anchor_path}")

            try:
                df = pd.read_csv(anchor_path, delimiter=r"\s+", header=0)
                sat_df = pd.read_csv(sat_path)
            except Exception as e:
                print(f"Error reading files {YEAR}{mm}{dd}{cy}: {e}")
                continue
            
            # convert numerical columns
            for col in ["vertco_reference_1@body", "obsvalue@body"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            for col in ["fg_depar@body", "an_depar@body", "biascorr_fg@body", "obsvalue@body", "peak_hPa"]:
                if col in sat_df.columns:
                    sat_df[col] = pd.to_numeric(sat_df[col], errors="coerce")

            # df = (df['obstype@hdr'] = 2 | df['obstype@hdr'] = 5)
            df['hPa'] = df['vertco_reference_1@body']/100.0
            # Initialize quantile columns
            for q in ["q25", "q50", "q75"]:
                sat_df[f"{q}_anchors"] = np.nan


            # Compute quantiles per pressure level
            for th in THRESHOLDS_HPA:
                while True:
                    layer_mask = (df["hPa"] >= th - layer_halfwidth) & (df["hPa"] <= th + layer_halfwidth)
                    subset = df.loc[layer_mask, "obsvalue@body"]
                    if not subset.empty: # if you found observations in this layer, stop here
                        break
                    if layer_halfwidth > 500:
                        print(f"NO ANCHORS for {th} hPa in {anchor_path}")
                        break
                    layer_halfwidth += layer_halfwidth  # expand the layer
                
                q25, q50, q75 = subset.quantile([0.25, 0.5, 0.75]).values
                mask = sat_df["peak_hPa"] == th
                if mask.any():
                    sat_df.loc[mask, ["q25_anchors", "q50_anchors", "q75_anchors"]] = [round(q25, 4), round(q50, 4), round(q75, 4)]

            sat_df.to_csv(out_path, index=False)
            print(f"Saved updated file: {out_path}")




