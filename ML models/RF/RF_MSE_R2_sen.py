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
channel = X_test['channel'].unique()
channels = np.sort(channel)
sensor_mse_data = []
sensor_r2_data = []
pred_mse_data = []
pred_r2_data = []
print(np.arange(len(pred_id_names)))

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

##################### MSE and R2 PER BIAS PREDICTOR:
for pred_id, pred_id_name in zip(pred_ids, pred_id_names):
    mask = (X_test[pred_id] == 1)
    if mask.sum() > 0:
        y_test_pred = y_test[mask]
        y_pred_pred = y_pred[mask]
        print("Bias predictor", pred_id_name)
        mse = round(mean_squared_error(y_test_pred, y_pred_pred), 4)
        print("MSE:", mse)
        r2 = r2_score(y_test_pred, y_pred_pred)
        print("r2:", r2)
    else:
        print("no value found for bias predictor:", pred_id_name)
        mse = 0
        r2 = 0
    pred_mse_data.append(mse)
    pred_r2_data.append(r2)

# MSE barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(pred_id_names))
ax.bar(x_pos, pred_mse_data, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(pred_id_names, fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
plt.ylim(0, max(pred_mse_data) * 1.1)
plt.tight_layout()
plt.savefig('MSE_pred.png')
plt.show()

# R2 barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(pred_id_names))
ax.bar(x_pos, pred_r2_data, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(pred_id_names, fontsize=16)
ax.set_ylabel('R2', fontsize=16)
plt.ylim(0, max(pred_r2_data) * 1.1)
plt.tight_layout()
plt.savefig('R2_pred.png')
plt.show()

##################### SENSORS:
for sensor, sensor_name in zip(sensors, sensor_names):
    mask = (X_test[sensor] == 1)
    if mask.sum() > 0:
        y_test_sensor = y_test[mask]
        y_pred_sensor = y_pred[mask]
        print("sensor", sensor_name)
        compute_r2(y_true = y_test_sensor, y_pred = y_pred_sensor)
        mse = round(mean_squared_error(y_test_sensor, y_pred_sensor), 4)
        print("MSE:", mse)
        r2 = r2_score(y_test_sensor, y_pred_sensor)
    else:
        print("no value found for sensor:", sensor)
        mse = 0
        r2 = 0
    sensor_mse_data.append(mse)
    sensor_r2_data.append(r2)

# MSE barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(sensor_names))
ax.bar(x_pos, sensor_mse_data, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(sensor_names)
ax.set_ylabel('MSE')
plt.ylim(0, max(sensor_mse_data) * 1.1)
plt.tight_layout()
plt.savefig('MSE_sensor.png')
plt.show()

# R2 barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(sensor_names))
ax.bar(x_pos, sensor_r2_data, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(sensor_names)
ax.set_ylabel('R2')
#ax.set_title('R2 per sensor')
plt.ylim(0, 1)  
plt.tight_layout()
plt.savefig('R2_sensor.png')
plt.show()
