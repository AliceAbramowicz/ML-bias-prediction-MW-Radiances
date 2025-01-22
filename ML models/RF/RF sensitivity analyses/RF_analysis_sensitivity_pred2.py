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
y_train = df_train["param"]
y_test = df_test["param"]

pickle_file = "RF_best_sensitivity_pred_2.pkl"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

best_RF = results_dict["best_estimator"]
cv_results = results_dict["cv_results"]
mse = results_dict["test_mse"]
r2 = results_dict["r2"]
feature_imp = results_dict["feature_importances"]
y_pred = results_dict["y_pred"]
print("R2:", r2)
print("MSE:", mse)

### FEATURE IMPORTANCE PLOT:
idx = feature_imp.argsort()
top_features_idx = idx[-18:]
plt.barh(X_train.columns[top_features_idx], feature_imp[top_features_idx])
plt.xlabel("feature importance")
plt.title("Random Forest: Feature Importance")
plt.tight_layout()
plt.show()
plt.close()

pred_ids = ['pred_id_0', 'pred_id_1', 'pred_id_2', 'pred_id_8', 'pred_id_9', 'pred_id_10']
pred_id_names = ['Constant', '1000-300hPa', '200-50hPa', 'nadir', 'nadir**2', 'nadir**3']
pred_colors = ["purple", "blue", "green", "orange", "red", "yellow"]
pred_mse_data = []
pred_r2_data = []

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
ax.bar(x_pos, pred_mse_data, color=pred_colors, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(pred_id_names)
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
plt.ylim(0, max(pred_mse_data) * 1.1)
plt.tight_layout()
plt.savefig('MSE_sensitivity_pred.png')
plt.show()

# R2 barplot:
fig, ax = plt.subplots()
x_pos = np.arange(len(pred_id_names))
ax.bar(x_pos, pred_r2_data, color=pred_colors, align='center')
ax.set_xticks(x_pos)
ax.set_xticklabels(pred_id_names)
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16)
ax.set_ylabel('R2', fontsize=16)
plt.ylim(min(pred_r2_data) * 1.1, 1)
plt.tight_layout()
plt.savefig('R2_sensitivity_pred.png')
plt.show()

### PREDICTION PLOT COLORED PER PRED_ID:
colors = []
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

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.5, c=colors)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
unique_colors = ["purple", "blue", "green", "orange", "red", "yellow"]
pred = {
        0: 'constant',
        1: '1000-300hPa thickness',
        2: '200-50hPa thickness',
        3: 'nadir view angle',
        4: 'nadir view angle **2',
        5: 'nadir view angle **3'
}
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Betas', fontsize=16)
plt.ylabel('Predicted Betas', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("Sensitivity_Pred2_Pred_VS_Test_per_pred.png")
plt.show()

### PREDICTION PLOT COLORED PER SENSOR:
# Sensors:
colors = []
for idx, row in X_test.iterrows():
    if row['sensor_3'] == 1:
        colors.append('purple')
    elif row['sensor_15'] == 1:
        colors.append('darkturquoise')
    elif row['sensor_73'] == 1:
        colors.append('orange')
    else:
        colors.append('black')

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.4, c=colors)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
unique_colors = ['purple', 'darkturquoise', 'orange']
plt.xlabel('True Betas', fontsize=16)
plt.ylabel('Predicted Betas', fontsize=16)
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
legend_handles = [Patch(facecolor=c, label=sensor_labels[i]) for i, c in enumerate(unique_colors)]
plt.legend(handles=legend_handles, loc='upper right', title='Sensors')
plt.grid(True)
plt.savefig('sensitivity_pred2_RF_ytest_VS_ypred_per_sensor.png')
plt.show()

