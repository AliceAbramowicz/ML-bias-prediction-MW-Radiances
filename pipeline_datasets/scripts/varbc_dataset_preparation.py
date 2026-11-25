#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import pandas as pd
import numpy as np

MAX_PRED = 6


def require_env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable: {name} "
                 f"(set it, or run via pipeline_dataset.py)")
    return value


def expand_pattern(raw_template, pattern, stamp):
    """Full path of the VARBC file for one timestamp.
    RAW_DATA_VARBC itself may contain {year}.
    """
    fields = {"stamp": stamp, "yyyy": stamp[0:4], "mm": stamp[4:6],
              "dd": stamp[6:8], "cy": stamp[8:10]}
    raw_dir = raw_template.replace("{year}", fields["yyyy"])
    return os.path.join(raw_dir, pattern.format(**fields))


def varbc_file_paths(raw_template, stamps, pattern):
    """Locate the VARBC files to parse.
    The layout below RAW_DATA_VARBC is given by varbc_pattern in the config,
    e.g. "{stamp}/VARBC.cycle.loop2"
    When $DATETIMES is set (a pipeline run) only the configured cycles are read.
    Standalone, with no $DATETIMES, the pattern is globbed instead.
    """
    if not stamps:
        if "{year}" in raw_template:
            sys.exit("RAW_DATA_VARBC contains {year} but $DATETIMES is not set. Run via pipeline_dataset.py or pass a concrete -i path")
        wildcard = pattern.format(stamp="*", yyyy="*", mm="*", dd="*", cy="*")
        return sorted(glob.glob(os.path.join(raw_template, wildcard)))

    paths = []
    n_missing = 0
    for stamp in stamps:
        path = expand_pattern(raw_template, pattern, stamp)
        if os.path.isfile(path):
            paths.append(path)
        else:
            print("Missing VARBC file: {}".format(path))
            n_missing += 1
    if n_missing:
        print("{} of {} VARBC files missing".format(n_missing, len(stamps)))
    return paths


def usage():
    print("Options:")
    print("  -i : directory holding the VARBC files ($RAW_DATA_VARBC)")
    print("  -o : output CSV path")
    print("  -h : Help")
    print("Layout below -i comes from $VARBC_PATTERN (varbc_pattern in the config)")

def readVARBC(file_path, valid_sats, valid_sensors):
    """
    Parse a single VARBC.cycle file and return a list of tuples:
    (ndata, key, npred, predcs, param0, params, predmean, predxcov)
    Only keeps entries matching selected satellites and sensors.
    """
    result=[]
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('ndata'):
                ndata = int(lines[i].split('=')[1].strip())
                if ndata > 1:
                    key = lines[i - 2].split('=')[1].strip()
                    try:
                        sat_str, sensor_str, channel = key.split()
                        sat = int(sat_str)
                        sensor = int(sensor_str)
                        if sensor not in valid_sensors or sat not in valid_sats:
                            continue  # Skip this entry if not matching the condition
                    except ValueError:
                        continue  # Skip incoherent entries

                    npred = int(lines[i + 1].split('=')[1].strip())
                    predcs = lines[i + 2].split('=')[1].strip()
                    param0 = lines[i + 3].split('=')[1].strip()
                    params = lines[i + 4].split('=')[1].strip()
                    predmean = lines[i + 7].split('=')[1].strip()
                    predxcnt = lines[i + 6].split('=')[1].strip()
                    predxcov = lines[i + 8].split('=')[1].strip()
                    result.append((ndata, key, npred, predcs, param0, params, predmean, predxcov))
        return(result)
 
def main():

    parser = argparse.ArgumentParser(
        description="Generate VARBC dataset csv",
        add_help=False)

    parser.add_argument("-i", "--input", default=os.environ.get("RAW_DATA_VARBC"),
                        help="Directory holding the VARBC files, {year} "
                             "placeholder allowed (default: $RAW_DATA_VARBC)")
    parser.add_argument("-o", "--output", default=os.environ.get("VARBC_OUTPUT_FILE"),
                        help="Path to save final CSV (default: $VARBC_OUTPUT_FILE)")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")

    args = parser.parse_args()
    
    if args.help:
        usage()
        sys.exit(0)

    VARBCINP = args.input

    if not VARBCINP or VARBCINP == "DUMMY":
        print("No VARBCINP directory provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)

    base_name = os.path.basename(os.path.normpath(VARBCINP))

    # Defined in the YAML
    VALID_SATS = [v.strip() for v in require_env("SATELLITES").split(",") if v.strip()]
    VALID_SENSORS = [v.strip() for v in require_env("SENSORS").split(",") if v.strip()]
    SAT_IDS = [int(v) for v in VALID_SATS]
    SENSOR_IDS = [int(v) for v in VALID_SENSORS]
    VARBC_PATTERN = require_env("VARBC_PATTERN")

    print("Starting to loop over files ...")
    print("VARBC layout: {}".format(VARBC_PATTERN))
    columns = ['time', 'sat', 'sensor', 'channel', 'ndata', 'npred', 'pred_id', 'param0', 'param', 'predmean']
    rows = []

    stamps = os.environ.get("DATETIMES", "").split()
    file_paths = varbc_file_paths(VARBCINP, stamps, VARBC_PATTERN)
    print("Parsing {} VARBC.cycle file(s)".format(len(file_paths)))

    # Go through the VARBC.cycle files
    for file_path in file_paths:
        YYYYMMDD = None
        HHMMSS_S = None
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                # 2nd line to get time
                if i == 1:
                    # store 2nd token in YYYYMMDD and 3rd in HHMMSS_S
                    YYYYMMDD, HHMMSS_S = line.split()[1:3]
                    break
        # If there's no date or cycle, skip this file
        if YYYYMMDD is None or HHMMSS_S is None:
            continue
        # transform time string to 6 int with leading 0's if necessary
        HHMMSS = "{:06d}".format(int(HHMMSS_S))
        
        result = readVARBC(file_path, SAT_IDS, SENSOR_IDS)
        if len(result) == 0:
            continue
        for tup in result:
            ndata, key, npred, predcs, param0, params, predmean, predxcov = tup
            name_key = '_'.join(key.split())
            
            sat, sensor, channel = key.split()
            time = "{}_{}".format(YYYYMMDD, HHMMSS)
            print(time)
            predxcov_values = [float(value) for value in predxcov.split()]
            predxcov_matrix = np.array(predxcov_values).reshape(npred, npred)

            for n in range(npred):
                # Collected in a list; the frame is built once below.
                rows.append({
                    'time': time,
                    'sat': sat,
                    'sensor': sensor,
                    'channel': channel,
                    'ndata': ndata,
                    'npred': npred,
                    'pred_id': int(predcs.split()[n]),
                    'param0': float(param0.split()[n]),
                    'param': float(params.split()[n]),
                    'predmean': float(predmean.split()[n]),
                    **{'predxcov_'+str(i+1): predxcov_matrix[i, n] for i in range(npred)}
                })

    varbc_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    # Convert and sort by time
    varbc_df['time_formatted'] = pd.to_datetime(varbc_df['time'], format='%Y%m%d_%H%M%S')
    varbc_df.sort_values(by='time_formatted', inplace=True)
    varbc_df.reset_index(drop=True, inplace=True)
    
    # Arrange varbc dataset here
    COV_COLUMNS_TO_KEEP = [f'predxcov_{i+1}' for i in range(MAX_PRED)]
    DROP_COLUMNS = ['time_formatted']
    
    varbc_df = varbc_df[varbc_df['sat'].astype(str).isin(VALID_SATS) & varbc_df['sensor'].astype(str).isin(VALID_SENSORS)]
    # Drop extra covariance columns
    cols_to_drop = [col for col in varbc_df.columns if col.startswith('predxcov_') and col not in COV_COLUMNS_TO_KEEP]
    cols_to_drop += [col for col in DROP_COLUMNS if col in varbc_df.columns]
    varbc_df.drop(columns=cols_to_drop, inplace=True)

    # the same columns in the same order for every domain, with
    # covariance columns absent from this run created as NaN. Without it a
    # 4-predictor dataset and a 6-predictor one produce different column sets,
    # which breaks the train/test alignment downstream.
    varbc_df = varbc_df.reindex(columns=columns + COV_COLUMNS_TO_KEEP)

    # Save output
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"{base_name}_common_sat_sen.csv"
    out_dir = os.path.dirname(output_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    varbc_df.to_csv(output_filename, header=True, index=False)
    print(f"Output available in", output_filename)


if __name__ == "__main__":
    main()