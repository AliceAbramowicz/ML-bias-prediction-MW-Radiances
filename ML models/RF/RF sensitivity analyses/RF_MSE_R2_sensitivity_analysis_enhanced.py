import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score

df_train = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2023.csv")
df_test = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2021.csv")
X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats.csv")
y_test = df_test["param"]
y_train = df_train["param"]

parser = argparse.ArgumentParser(description="Process pickle file and generate MSE and R2 bar plots.")
parser.add_argument("pickle_file", type=str, help="Path to the pickle file")
args = parser.parse_args()

pickle_filename = args.pickle_file.split('/')[-1].split('.')[0]

with open(args.pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

best_RF = results_dict["best_estimator"]
cv_results = results_dict["cv_results"]
mse = results_dict["test_mse"]
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
sensor_names = ['MWHS2', 'AMSU-A', 'MHS']
sensor_colors = ['orange', 'purple', 'darkturquoise']
pred_ids = ['pred_id_0', 'pred_id_1', 'pred_id_2', 'pred_id_8', 'pred_id_9', 'pred_id_10']
pred_id_names = ['Constant', '1000-300hPa', '200-50hPa', 'nadir', 'nadir**2', 'nadir**3']
channel = X_test['channel'].unique()
mse_data = []
r2_data = []

##################### PREDICTOR BIAS - SENSORS - CHANNEL:
for sensor, sensor_name in zip(sensors, sensor_names):
    sensor_mse_data = []
    sensor_r2_data = [] 
    mask = (X_test[sensor] == 1)
    if mask.sum() > 0:
        y_test_sensor_channel = y_test[mask]
        y_pred_sensor_channel = y_pred[mask]
        print("sensor", sensor_name)
        mse = round(mean_squared_error(y_test_sensor_channel, y_pred_sensor_channel), 4)
        print("MSE:", mse)
        r2 = r2_score(y_test_sensor_channel, y_pred_sensor_channel)
        print("R2:", r2)
    else:
        print("no value found for sensor:", sensor, "channel ", channel)
        mse = 0
        r2 = 0
    mse_data.append(mse)
    r2_data.append(r2)

# MSE barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(sensor_names))
colors = sensor_colors

ax.bar(x_pos, mse_data, color=colors, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(sensor_names, fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, max(mse_data) * 1.1)
plt.tight_layout()
plt.savefig(f'MSE_{pickle_filename}.png')
plt.show()

# R2 barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(sensor_names))
colors = sensor_colors
ax.bar(x_pos, r2_data, color=colors, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(sensor_names, fontsize=16)
ax.set_ylabel('R2', fontsize=16)
plt.ylim(0, 1)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f'R2_{pickle_filename}.png')
plt.show()


