import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score

df_train = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2023.csv")
df_test = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2021.csv")
X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats.csv")
y_test = df_test["param"]
y_train = df_train["param"]

pickle_file = "RF_best_result.pkl"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

best_RF = results_dict["best_estimator"]
cv_results = results_dict["cv_results"]
mse = results_dict["mse"]
r2 = results_dict["r2"]
feature_imp = results_dict["feature_importances"]
y_pred = results_dict["y_pred"]

num_nans = np.isnan(y_pred).sum()
print("count of NAs in y pred:", num_nans, len(y_pred))
num_nans = np.isnan(y_test).sum()
print("count of NAs in y test:", num_nans, len(y_test))

y_test = y_test.to_numpy()
y_train = y_train.to_numpy()

sensors = ['sensor_73', 'sensor_3', 'sensor_15']
sensor_names = ['MWHS2', 'AMSUA', 'MHS']
pred_ids = ['pred_id_0', 'pred_id_1', 'pred_id_2', 'pred_id_8', 'pred_id_9', 'pred_id_10']
pred_id_names = ['Constant', '1000-300hPa', '200-50hPa', 'nadir', 'nadir**2', 'nadir**3']
sats = ['sat_3','sat_5','sat_209','sat_223','sat_523']
sat_names = ['Metop B', 'Metop C','NOAA18', 'NOAA19', 'FY-3D']
channel = X_test['channel'].unique()
channels = np.sort(channel)
mse_data = []
r2_data = []

def compute_r2(y_true, y_pred):
    y_mean = np.mean(y_true)
    # Calculate the total sum of squares (SST)
    ss_total = np.sum((y_true - y_mean) ** 2)
    # Calculate the residual sum of squares (SSR)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    #print(f"SSR: {ss_residual}")
    #print(f"SSR/SST: {ss_residual/ss_total}")
    print(f"R²", round(r2, 3))


##################### PREDICTOR BIAS - SENSORS - CHANNEL:
for sensor, sensor_name in zip(sensors, sensor_names):
    sensor_mse_data = []
    sensor_r2_data = []
    for sat, sat_name in zip(sats, sat_names):
        mse_by_sat = []
        r2_by_sat = []
        for pred_id, pred_id_name in zip(pred_ids, pred_id_names):
            mse_by_channel = []
            r2_by_channel = []
            for channel in channels:
                mask = (X_test[sensor] == 1) & (X_test[sat] == 1) & (X_test['channel'] == channel) & (X_test[pred_id] == 1)
                if mask.sum() > 0:
                    y_test_sensor_channel = y_test[mask]
                    y_pred_sensor_channel = y_pred[mask]
                    print("sensor", sensor_name, "channel", channel)
                    compute_r2(y_true = y_test_sensor_channel, y_pred = y_pred_sensor_channel)
                    mse = round(mean_squared_error(y_test_sensor_channel, y_pred_sensor_channel), 4)
                    print("MSE:", mse)
                    r2 = r2_score(y_test_sensor_channel, y_pred_sensor_channel)
                else:
                    print("no value found for sensor:", sensor, "channel ", channel)
                    mse = 0
                    r2 = 0

                mse_by_channel.append(mse)
                r2_by_channel.append(r2)

            mse_by_sat.append(mse_by_channel)
            r2_by_sat.append(r2_by_channel)
        sensor_mse_data.append(mse_by_sat)
        sensor_r2_data.append(r2_by_sat)
    mse_data.append(sensor_mse_data)
    r2_data.append(sensor_r2_data)
        
    # mse_data is a nested list
    mse_data.append(mse_by_sat)
    print("mse_data:", mse_data)
    print("mse_by_channel:", mse_by_channel)
    print("sensor_mse_data:", sensor_mse_data)
    r2_data.append(r2_by_sat)

channel_values = X_test["channel"]
norm = plt.Normalize(min(channel_values), max(channel_values))

# MSE barplot:
# pair each sensor name with its corresponding MSE and enumerate with i through each of the 3 pairs
for i, (sensor_name, sensor_mse_data) in enumerate(zip(sensor_names, mse_data)):
    for j, (sat_name, sat_mse_data) in enumerate(zip(sat_names, sensor_mse_data)):
        fig, ax = plt.subplots(figsize=(10, 10))
        bar_width = 0.1
        indices = np.arange(len(channels))
        for k, (pred_name, mse_by_channel) in enumerate(zip(pred_id_names,sat_mse_data)):
            ax.bar(x=indices + k * bar_width, height=mse_by_channel, width=bar_width, label=pred_name)

    ax.set_xticks(indices + bar_width * (len(pred_id_names) - 1) / 2)
    ax.set_xticklabels(channels)
    ax.set_xlabel('Channels', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_title(f'{sensor_name} {sat_name}: MSE per channel & predictor')
    ax.legend(title='Predictor', fontsize=16)
    plt.savefig('RF_MSE_Sat_Sen_Chan.png')
    plt.show()

# R2 barplot:
for i, (sensor_name, sensor_r2_data) in enumerate(zip(sensor_names, r2_data)):
    print("i:", i)
    fig, ax = plt.subplots(figsize=(10, 10))
    bar_width = 0.1
    indices = np.arange(len(channels))
    for j, (sat_name, r2_by_channel) in enumerate(zip(sat_names,r2_data[i])):
        print("j:", j)
        ax.bar(x=indices + j * bar_width, height=r2_by_channel, width=bar_width, label=sat_name)

    ax.set_xticks(indices + bar_width * (len(sat_names) - 1) / 2)
    ax.set_xticklabels(channels)
    ax.set_xlabel('Channels')
    ax.set_ylabel('R2')
    ax.set_title(f'{sensor_name} R2 per channel & bias predictor')
    ax.legend(title='Sensors')
    plt.savefig('RF_R2_Sat_Sen_Chan.png')
    plt.show()

