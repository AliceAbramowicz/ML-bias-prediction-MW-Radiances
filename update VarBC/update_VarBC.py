import sys
import os
import getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

def usage():
    print("Options:")
    print("  -i : VARBCINP directory")
    print("  -d : DATASET file")
    print("  -p : PICKLE file")
    print("  -m : ML model")
    print("example: \nroot_dir = /perm/nld3863/pipeline_datasets/datasets/VarBC_2021 \ndf: /perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2021.csv \npickle_file = /perm/nld3863/FINAL_ML_MODELS/RF/RF_results.pkl, \nML model: RF")

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:d:p:m:h")
except getopt.GetoptError as err:
    print(str(err))
    usage()
    sys.exit(1)

for opt, arg in opts:
    if opt == '-h':
        usage()
        sys.exit(0)
    elif opt == '-i':
        VARBCINP = arg
    elif opt == '-d':
        DATASET = arg
    elif opt == '-p':
        PICKLE = arg
    elif opt == '-m':
        MODEL = arg

if VARBCINP == "DUMMY":
        print("No VARBCINP directory provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)

if DATASET == 'DUMMY':
    print('no dataset provided.')
    print("Try '{} -h' for more information".format(sys.argv[0]))
    sys.exit(1)

if PICKLE == 'DUMMY':
    print('no params results provided (pickle format).')
    print("Try '{} -h' for more information".format(sys.argv[0]))
    sys.exit(1)

if MODEL == 'DUMMY':
    print('no ML model type specified. This helps to keep the data tidy.')
    print("Try '{} -h' for more information".format(sys.argv[0]))
    sys.exit(1)

### import my dataset and predictions
df = pd.read_csv(DATASET)
pickle_file = PICKLE

y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

if "y_pred_scaled" in results_dict:
    y_pred_scaled = results_dict["y_pred_scaled"]
    y_pred_original_scale = y_scaler.inverse_transform(y_pred_scaled)
else:
    y_pred_original_scale = results_dict["y_pred"]

df.drop(columns=['param', 'param0', 'cycle'], inplace=True)

time = pd.to_datetime(df['time'], format='%Y%m%d_%H%M%S')
df['time_formatted'] = pd.to_datetime(df['time'], format='%Y%m%d_%H%M%S')
df['date'] = time.dt.strftime('%Y%m%d').astype(str)
df['cycle'] = time.dt.strftime('%H%M%S').astype(str)
df['params'] = y_pred_original_scale.astype(float)
df['params'] = df['params'].apply(lambda x: f"{x:.3E}")
df = df.sort_values(by=['time_formatted'])
df['pred_id'] = df['pred_id'].astype(int)


df_prev = df[['sat', 'sensor', 'channel', 'pred_id', 'time_formatted', 'params']].copy()
df_prev['time_formatted'] = df_prev['time_formatted'] + pd.Timedelta(hours=24)
df_prev.rename(columns={'params': 'param0'}, inplace=True)

df = pd.merge(
    df,
    df_prev,
    on=['sat', 'sensor', 'channel', 'pred_id', 'time_formatted'],
    how='left'
)
#  SAME RESULTS:
# df['param0'] = df.groupby(['sat', 'sensor', 'channel', 'pred_id', 'cycle'])['params'].shift(periods=1, fill_value=np.nan)
# some datagroups might not have data at some cycles depending on the day, so cannot shift by 8 rows (i.e. 24/3=8). Find a way to shift by 24h
df = df.sort_values(by=['time_formatted', 'sat', 'sensor', 'channel', 'pred_id'])

df = df.astype(str)
print(df.columns)
print(df[['date', 'cycle', 'sat', 'sensor', 'channel', 'pred_id', 'params', 'param0']].tail(150))

df.to_csv('DF_with_param0.csv', index=False)

if not os.path.exists(VARBCINP):
    raise ValueError(f"Folder does not exist: {VARBCINP}")

for root, dirs, files in os.walk(VARBCINP):
        for file in files:
            if file == 'VARBC.cycle':
                file_path = os.path.join(root, file) # traditional VARBC.cycle files
                print(file_path)
                YYYYMMDD = None
                HHMMSS_S = None
                temp_file = file_path + "_" + MODEL

                with open(file_path, "r") as f, open(temp_file, "w") as out_f:
                
                    for i, line in enumerate(f):
                        if i == 1:
                            YYYYMMDD, HHMMSS_S = line.split()[1:3]
                            HHMMSS = "{:06d}".format(int(HHMMSS_S))
                            print('start processing file file_path')
                            print('YYYYMMDD', YYYYMMDD)
                            print('HHMMSS', HHMMSS)
                            break
                    
                    f.seek(0)

                    sat = sensor = channel = None
                    new_param0 = ""
                    new_params = ""
                    prev_prev_line = prev_line = None
                    zero_out = False

                    for line in f:
                        if line.startswith('ndata'):
                            ndata_value = int(line.split('=')[1].strip())
                        
                            key = prev_prev_line.split('=')[1].strip()
                            sat, sensor, channel = key.split()[:3]
                            print('sat:', sat, 'sensor:', sensor, 'channel:', channel)

                            if sat in ['4', '206']:
                                ndata_value = 0
                                print(f"wrong satellite ({sat}) so set ndata=0")

                            if ndata_value >= 1:

                                matching_rows = df[
                                    (df['date'] == YYYYMMDD) &
                                    (df['cycle'] == HHMMSS) &
                                    (df['sat'] == sat) &
                                    (df['sensor'] == sensor) &
                                    (df['channel'] == channel)]

                                print(f"Filtering with: YYYYMMDD={YYYYMMDD}, HHMMSS={HHMMSS}, sat={sat}, sensor={sensor}, channel={channel}")
                                print(f"Matching rows found: {len(matching_rows)}")

                                if not matching_rows.empty:
                                    print("CONDITION FULFILLED:", matching_rows)
                                    # order matching_rows based on pred_id:
                                    df['pred_id'] = df['pred_id'].astype(int) # if you keep it as a string, 10 is considered smaller than 8
                                    matching_rows = matching_rows.sort_values(by='pred_id')

                                    new_param0 = " ".join(map(str, matching_rows['param0']))
                                    new_params = " ".join(map(str, matching_rows['params']))

                                else:
                                    print("ndata positive but not in the dataset. Set it to 0.")
                                    zero_out = True
                                    ndata_value = 0

                            if ndata_value == 0:
                                    print("ndata 0. Set its params to 0.")
                                    zero_out = True
                                    ndata_value = 0

                            line = f"ndata={ndata_value}\n"

                        elif line.startswith("param0="):
                            if zero_out:
                                # count = len(line.split()) - 1  # exclude 'param0='
                                values_part = line[len("param0="):].strip()
                                count = len(values_part.split())
                                new_param0 = "  ".join(["0.000E+00"] * count)
                            if new_param0:
                                line = f"param0= {new_param0}\n"
                                new_param0 = ""


                        elif line.startswith("params="):
                            if zero_out:
                                values_part = line[len("params="):].strip() # count number of entries without 'params=' because sometimes there is no space
                                count = len(values_part.split()) 
                                new_params = "  ".join(["0.000E+00"] * count)
                            if new_params:
                                line = f"params= {new_params}\n"
                                new_params = ""
                                zero_out = False
                    
                        out_f.write(line)

                        prev_prev_line, prev_line = prev_line, line
                
                    print(f"File {file_path} modified.")


