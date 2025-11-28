#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os, sys, json

# -----------------------------
# Environment setup
# -----------------------------

try:
    SAT_TABLES_DIR = os.environ["SAT_TABLES_DIR"]
    DOMAIN = os.environ["DOMAIN"]
    YEAR = os.environ["YEAR"]
    DAYS_JSON = json.loads(os.environ.get("DAYS_JSON", "{}"))
    MONTHS = os.environ.get("MONTHS", "").split()
    CYCLES = os.environ.get("CYCLES", "").split()
except KeyError as e:
    sys.exit(f"Missing environment variable: {e}")

SENSORS = os.environ.get("SENSORS", "3,15,73").split(",")

print(f"Reading input from: {SAT_TABLES_DIR}")
print(f"Writing outputs to: {SAT_TABLES_DIR}\n")

# create for each sensor a dict with correspondance between chan and weight function peak height (hPa)
chan_hpa_amsua = {
        '6': 400,
        '7': 270,
        '8': 180,
        '9': 90
        }

chan_hpa_mhs = {
        '3': 400,
        '4': 600,
        '5': 800
}

chan_hpa_mwhs2 = {
        '4': 100, 
        '5': 250, 
        '6': 300, 
        '11': 350, 
        '12': 400, 
        '13': 500, 
        '14': 550, 
        '15': 650
}

# Sensor-to-dictionary mapping
chan_hpa_dicts = {
    '3': chan_hpa_amsua,
    '15': chan_hpa_mhs,
    '73': chan_hpa_mwhs2
}

# -----------------------------
# Main processing loop
# -----------------------------
for mm in MONTHS:
    days = [f"{int(d):02d}" for d in DAYS_JSON.get(str(mm).zfill(2), [])]
    for dd in days:
          for cy in CYCLES:
                in_path = f"{SAT_TABLES_DIR}/sat_{YEAR}{mm}{dd}{cy}.txt"
                out_path = f"{SAT_TABLES_DIR}/sat_{YEAR}{mm}{dd}{cy}_with_hPa.csv"
                if os.path.exists(out_path):
                        print(f"Skipping {out_path} — already exists.")
                        continue
                if not os.path.exists(in_path):
                        print(f"Skipping missing file: {in_path}")
                        continue
                if os.path.getsize(in_path) < 100:
                        print(f"Skipping empty file: {in_path}")
                        continue
                
                df = pd.read_csv(in_path, delimiter=r"\s+", header=0)
                print(f"Processing: {in_path}")
                
                #  for all sensors
                for s in SENSORS:
                        print(f"Checking sensor: {s}")
                        mask = df["sensor@hdr"].astype(str).str.contains(s)
                        if not mask.any():
                                print(f"No rows found for {s}")
                                continue
                        chan_dict = chan_hpa_dicts.get(s, {})

                        # find hPa value corresponding to chan and create new column 'peak_hPa'
                        df.loc[mask, "peak_hPa"] = df.loc[mask, "vertco_reference_1@body"].astype(str).map(chan_dict)

                # Save updated file
                df.to_csv(out_path, index=False)
                print(f"Saved updated file: {out_path}")





