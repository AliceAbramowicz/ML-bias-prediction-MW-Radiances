import sys
import getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")

def usage():
    print("Options:")
    print(" -m : model (e.g. RF, AB, NN)")
    print(" -s : sensor excluded from training (e.g. amsua, mhs, mwhs2)")
    print(" -p : plots per (e.g. bias_preds, sat, sensors)")
    print(" -i : initial results of your ML model when all sensors are included in training (e.g. /perm/nld3863/FINAL_ML_MODELS/RF/RF_final.pkl)")

try:
    opts, args = getopt.getopt(sys.argv[1:], "m:s:p:i:h")
except getopt.GetoptError as err:
    print(str(err))
    usage()
    sys.exit(1)

for opt, arg in opts:
    if opt == '-h':
        usage()
        sys.exit(0)
    elif opt == '-m':
        model = arg
    elif opt == '-s':
        SENSOR = arg
    elif opt == '-p':
        plots = arg
    elif opt == '-i':
        initial_pickle_file = arg

if model == "DUMMY":
    print("No model specified, try -h for more info")
    sys.exit(1)

if SENSOR == "DUMMY":
    print("no sensor specified, try -h for more info")
    sys.exit(1)

if initial_pickle_file == "DUMMY":
    initial_pickle_file = '/perm/nld3863/FINAL_ML_MODELS/RF/RF_final.pkl'

pickle_file = f"{model}/{model}_sensitivity_{SENSOR}.pkl"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

if (model == 'AB') or (model == 'RF'):
    # results of ML model with all data included in training
    with open(initial_pickle_file, 'rb') as f:
        results_initial = pickle.load(f)
    y_pred_initial = results_initial["y_pred"]

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    best_model = results_dict["best_estimator"]
    print(best_model)
    mse = results_dict["test_mse"]
    r2 = results_dict["r2"]
    y_pred = results_dict["y_pred"]
    print(best_model)
    print("mse:", mse, "r2:", r2)
    print('y_pred length:', len(y_pred))
    print('y_test length:', len(y_test))

    ### FEATURE IMPORTANCE PLOT:
    feature_imp = results_dict["feature_importances"]
    idx = feature_imp.argsort()
    top_features_idx = idx[-18:]
    plt.barh(X_train.columns[top_features_idx], feature_imp[top_features_idx])
    plt.xlabel("feature importance")
    plt.title("Random Forest: Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{model}/{model}_{SENSOR}_feature_importance.png')
    plt.close()

elif (model == 'NN'):
    # results of ML model with all data included in training
    with open(initial_pickle_file, 'rb') as f:
        results_initial = pickle.load(f)
    y_pred_initial = results_initial["y_pred_scaled"]

    # import tensorflow as tf
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    best_model = results_dict["final_model_path"]
    # model = tf.keras.models.load_model(f'NN/{best_model}')
    # print("NN model's architecture:")
    # model.summary()

    # Scale your results back 
    y_pred_scaled = results_dict["y_pred_scaled"]
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_pred_initial = y_scaler.inverse_transform(y_pred_initial)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("mse:", mse, "r2", r2)

### CATEGORIES
colors = []
# Bias predictors:
if plots == 'bias_preds':
    for idx, row in X_test.iterrows():
        if row['pred_id_0'] == 1:
            colors.append('purple')
        elif row['pred_id_1'] == 1:
            colors.append('blue')
        elif row['pred_id_2'] == 1:
            colors.append('green')
        elif row['pred_id_8'] == 1:
            colors.append('orange')
        elif row['pred_id_9'] == 1:
            colors.append('red')
        elif row['pred_id_10'] == 1:
            colors.append('yellow')
        else:
            colors.append('black')

# Satellites:
elif plots == 'sat':
    for idx, row in X_test.iterrows():
        if row['sat_3'] == 1:
            colors.append('purple')
        elif row['sat_5'] == 1:
            colors.append('blue')
        elif row['sat_209'] == 1:
            colors.append('green')
        elif row['sat_223'] == 1:
            colors.append('orange')
        elif row['sat_523'] == 1:
            colors.append('red')
        else:
            colors.append('black')

# Sensors:
elif plots == 'sensors':
    for idx, row in X_test.iterrows():
        if row['sensor_3'] == 1:
            colors.append('purple')
        elif row['sensor_15'] == 1:
            colors.append('darkturquoise')
        elif row['sensor_73'] == 1:
            colors.append('orange')
        else:
            colors.append('black')

### PREDICTION PLOT COLORED PER SENSOR:
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c=colors)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
plt.xlabel('True Betas')
plt.ylabel('Predicted Betas')
unique_colors = ["purple", "darkturquoise", "orange"]
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
legend_handles = [Patch(facecolor=c, label=sensor_labels[i]) for i, c in enumerate(unique_colors)]
#plt.legend(handles=legend_handles, loc='lower right', title='Sensors')
#plt.title(f"{model} removing {sensor} from training (MSE: {round(mse, 4)})")
plt.xlabel('True Betas', fontsize=15)
plt.ylabel('Predicted Betas', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.grid(True)
plt.savefig(f'{model}/sensitivity_{SENSOR}_{model}_ytest_VS_ypred.png')
plt.show()

### BARPLOT FOR MSE & R2 OF DIFFERENT SENSORS
sensors = ['sensor_73', 'sensor_3', 'sensor_15']
sensor_names = ['MWHS2', 'AMSU-A', 'MHS']
sensor_colors = ['orange', 'purple', 'darkturquoise']
pred_ids = ['pred_id_0', 'pred_id_1', 'pred_id_2', 'pred_id_8', 'pred_id_9', 'pred_id_10']
pred_id_names = ['Constant', '1000-300hPa', '200-50hPa', 'nadir', 'nadir**2', 'nadir**3']
mse_data_initial = []
mse_data_sensitivity = []
r2_data_initial = []
r2_data_sensitivity = []

# Compute MSE and R2 for each sensor in both models
for sensor, sensor_name in zip(sensors, sensor_names):
    mask = (X_test[sensor] == 1)
    if mask.sum() > 0:
        # MSE and R2 for the initial model
        y_test_sensor = y_test[mask]
        y_pred_initial_sensor = y_pred_initial[mask]
        mse_initial = round(mean_squared_error(y_test_sensor, y_pred_initial_sensor), 4)
        mse_data_initial.append(mse_initial)
        r2_initial = round(r2_score(y_test_sensor, y_pred_initial_sensor), 4)
        r2_data_initial.append(r2_initial)

        # MSE and R2 for the sensitivity model
        y_pred_sensor = y_pred[mask]
        mse_sensitivity = round(mean_squared_error(y_test_sensor, y_pred_sensor), 4)
        mse_data_sensitivity.append(mse_sensitivity)
        r2_sensitivity = round(r2_score(y_test_sensor, y_pred_sensor), 4)
        r2_data_sensitivity.append(r2_sensitivity)
        print(f"Sensor: {sensor_name}, Initial MSE: {mse_initial}, Sensitivity MSE: {mse_sensitivity}")
        print(f"Sensor: {sensor_name}, Initial MSE: {mse_initial}, Sensitivity MSE: {r2_sensitivity}")

    else:
        print(f"No value found for sensor: {sensor_name}")
        mse_data_initial.append(0)
        mse_data_sensitivity.append(0)
        r2_data_initial.append(0)
        r2_data_sensitivity.append(0)
# MSE barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(sensor_names))
width = 0.35
# Plot bars for initial model (in black)
ax.bar(x_pos - width/2, mse_data_initial, width, label='Initial Model', color='black')
# Plot bars for sensitivity model (in respective sensor colors)
ax.bar(x_pos + width/2, mse_data_sensitivity, width, label='Sensitivity Model', color=sensor_colors)

ax.set_xticks(x_pos)
ax.set_xticklabels(sensor_names, fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, max(max(mse_data_initial), max(mse_data_sensitivity)) * 1.1)
plt.tight_layout()
plt.savefig(F'{model}/MSE_sensors_sensitivity_{SENSOR}.png')
plt.show()

#R2 barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(sensor_names))
width = 0.35
# Plot bars for initial model (in black)
ax.bar(x_pos - width/2, r2_data_initial, width, label='Initial Model', color='black')
# Plot bars for sensitivity model (in respective sensor colors)
ax.bar(x_pos + width/2, r2_data_sensitivity, width, label='Sensitivity Model', color=sensor_colors)
ax.set_xticks(x_pos)
ax.set_xticklabels(sensor_names, fontsize=16)
ax.set_ylabel('R2', fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{model}/R2_sensors_sensitivity_{SENSOR}.png')
plt.show()



### PERFORMANCES PER SENSOR:
# sen_dict = {'amsua': 'sensor_3', 'mhs': 'sensor_15', 'mwhs2': 'sensor_73'}

# for key in sen_dict:
#     print(key)
#     mask = X_test[sen_dict[key]] == 1
#     y_test_{key} = y_test[mask]
#     f'y_pred_{key}' = y_pred[mask]
#     f'X_test_{key}' = X_test[mask]
#     f'mse_{key}' = mean_squared_error(f'y_test_{key}', f'y_pred_{key}')
#     f'r2_{key}' = r2_score(f'y_test_{key}', f'y_pred_{key}')

#     print("total mse:", mse)
#     print(f"mse {key}:", f'mse_{key}')
#     print("total r2:", r2)
#     print(f"r2 {key}:", f"r2_{key}")

# isolate MWHS2 sensor only:
mask = X_test['sensor_73'] == 1
y_test_mwhs2 = y_test[mask]
y_pred_mwhs2 = y_pred[mask]
X_test_mwhs2 = X_test[mask]
mse_mwhs2 = mean_squared_error(y_test_mwhs2, y_pred_mwhs2)
r2_mwhs2 = r2_score(y_test_mwhs2, y_pred_mwhs2)

# isolate AMSUA sensor only:
mask = X_test['sensor_3'] == 1
y_test_amsua = y_test[mask]
y_pred_amsua = y_pred[mask]
X_test_amsua = X_test[mask]
mse_amsua = mean_squared_error(y_test_amsua, y_pred_amsua)
r2_amsua = r2_score(y_test_amsua, y_pred_amsua)

# isolate MHS sensor only:
mask = X_test['sensor_15'] == 1
y_test_mhs = y_test[mask]
y_pred_mhs = y_pred[mask]
X_test_mhs = X_test[mask]
mse_mhs = mean_squared_error(y_test_mhs, y_pred_mhs)
r2_mhs = r2_score(y_test_mhs, y_pred_mhs)

print("total mse:", mse)
print("mse mwhs2:", mse_mwhs2)
print("mse amsua:", mse_amsua)
print("mse mhs:", mse_mhs)
print("total r2:", r2)
print("r2 mwhs2:", r2_mwhs2)
print("r2 amsua:", r2_amsua)
print("r2 mhs:", r2_mhs)