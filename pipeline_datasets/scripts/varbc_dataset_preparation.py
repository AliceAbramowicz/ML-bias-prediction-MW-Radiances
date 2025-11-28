#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import numpy as np

def usage():
    print("Options:")
    print("  -i : VARBCINP directory")
    print("  -h : Help")

def readVARBC(file_path):
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
                        if sensor not in [3, 15, 73] or sat not in [3, 5, 209, 223, 523]:
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

    parser.add_argument("-i", "--input", required=True, help="Path to VARBCINP directory")
    parser.add_argument("-o", "--output", required=False, help="Path to save final CSV (optional)")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")

    args = parser.parse_args()
    
    if args.help:
        usage()
        sys.exit(0)

    VARBCINP = args.input
    base_name = os.path.basename(os.path.normpath(VARBCINP))
    

    if VARBCINP == "DUMMY":
        print("No VARBCINP directory provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)

    print("Starting to loop over files ...")
    columns = ['time', 'sat', 'sensor', 'channel', 'ndata', 'npred', 'pred_id', 'param0', 'param', 'predmean']
    varbc_df = pd.DataFrame(columns=columns)

    # Go through all files within directory VARBCINP
    for root, dirs, files in os.walk(VARBCINP):
        for file in files:
            if file == "VARBC.cycle":
                # join root directory "root" to the file called VARBC.cycle
                file_path = os.path.join(root, file)
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
                
                result = readVARBC(file_path)
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
                        new_row = pd.DataFrame([{
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
                        }])
                        
                        varbc_df = pd.concat([varbc_df, new_row], ignore_index=True)

    # Convert and sort by time
    varbc_df['time_formatted'] = pd.to_datetime(varbc_df['time'], format='%Y%m%d_%H%M%S')
    varbc_df.sort_values(by='time_formatted', inplace=True)
    varbc_df.reset_index(drop=True, inplace=True)
    
    # Arrange varbc dataset here
    VALID_SATS = ['3', '5', '209', '223', '523']
    VALID_SENSORS = ['3', '15', '73']
    MAX_PRED = 6
    COV_COLUMNS_TO_KEEP = [f'predxcov_{i+1}' for i in range(MAX_PRED)]
    DROP_COLUMNS = ['time_formatted']
    
    varbc_df = varbc_df[varbc_df['sat'].astype(str).isin(VALID_SATS) & varbc_df['sensor'].astype(str).isin(VALID_SENSORS)]
    # Drop extra covariance columns
    cols_to_drop = [col for col in varbc_df.columns if col.startswith('predxcov_') and col not in COV_COLUMNS_TO_KEEP]
    cols_to_drop += [col for col in DROP_COLUMNS if col in varbc_df.columns]
    varbc_df.drop(columns=cols_to_drop, inplace=True)

    # Save output
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"{base_name}_common_sat_sen.csv"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    varbc_df.to_csv(output_filename, header=True, index=False)
    print(f"Output available in", output_filename)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No command line arguments provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)
    main()
