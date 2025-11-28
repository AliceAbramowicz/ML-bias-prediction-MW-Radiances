import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv('/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv')
y_test = pd.read_csv('/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv')

###### If you have results ready:
pickle_file = "RF_final.pkl"
with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

best_RF = results_dict["best_estimator"]
mse = results_dict["mse"]
r2 = results_dict["r2"]
feature_imp = results_dict["feature_importances"]
y_pred = results_dict["y_pred"]
print('RF model', best_RF)

y_test = y_test.to_numpy().ravel()
y_train = y_train.to_numpy().ravel()

###### If you must run your RF model here:
# RF = RandomForestRegressor(max_features=28, min_samples_leaf=4, n_estimators=500, random_state=100) # results found in hyperparam grid search
# model = RF.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# feature_imp = model.feature_importances_
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# results_dict = {
#     'best_estimator': RF, 
#     'feature_importances': feature_imp,
#     'mse': mse, 
#     'r2': r2,
#     'y_pred': y_pred
#     }

# print(results_dict)

# pickle_file = 'RF_final.pkl'

# with open(pickle_file, 'wb') as f:
#     pickle.dump(results_dict, f)

# print("RF results saved in RF_final.pkl")

print('mse:', mse)
print('r2:', r2)

### FEATURE IMPORTANCE PLOT:
idx = feature_imp.argsort()
top_features_idx = idx[-18:]
plt.figure(figsize=(6, 4))
plt.barh(X_train.columns[top_features_idx], feature_imp[top_features_idx])
plt.xlabel("feature importance")
plt.tight_layout()
plt.savefig('Feature_importance.png')
plt.show()
plt.close()

colors = []

### PERFORMANCES PER SENSOR:
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


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c=colors)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlabel('True Betas', fontsize=15)
plt.ylabel('Predicted Betas', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
unique_colors = ["purple", "darkturquoise", "orange"]
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
legend_handles = [Patch(facecolor=c, label=sensor_labels[i]) for i, c in enumerate(unique_colors)]
# plt.legend(handles=legend_handles, loc='lower right', title='Sensors')
plt.grid(True)
plt.tight_layout()
plt.savefig('RF_ytest_VS_ypred_colored_sensors.png')
plt.show()

### PREDICTION PLOT COLORED PER BIAS PREDICTOR:
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

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, c=colors)
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlabel('True Betas', fontsize=15)
plt.ylabel('Predicted Betas', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.grid(True)
plt.savefig('RF_ytest_VS_ypred_colored_pred.png')
plt.show()

### PREDICTIONS plot AMSUA
colors=[]
for idx, row in X_test_amsua.iterrows():
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

plt.figure(figsize=(10, 10))
scatter = plt.scatter(y_test_amsua, y_pred_amsua, c=colors, linewidths=1, alpha=0.5)
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
sat_labels = {
        0: 'METOP B',
        1: 'METOP C',
        2: 'NOAA 18',
        3: 'NOAA 19',
        4: 'FY-3D'
}
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
legend_handles = [Patch(facecolor=c, label=pred[i]) for i, c in enumerate(unique_colors)]
plt.legend(handles=legend_handles, loc='lower right', title='Predictors')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xticks()
plt.yticks()
plt.xlabel('True Betas')
plt.title('AMSUA: Predicted VS True Betas (MSE: {:.4f})'.format(mse_amsua))
# plt.savefig("AMSUA_Pred_VS_Test.png")
plt.show()

### PREDICTIONS plot MHS:
colors = []

for idx, row in X_test_mhs.iterrows():
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

plt.figure(figsize=(8, 6))
scatter = plt.scatter(y_test_mhs, y_pred_mhs, c=colors, linewidths=1, alpha=0.5)
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
legend_handles = [Patch(facecolor=c, label=pred[i]) for i, c in enumerate(unique_colors)]
plt.legend(handles=legend_handles, loc='lower right', title='Predictors')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Betas')
plt.title('MHS: Predicted VS True Betas (MSE: {:.4f})'.format(mse_mhs))
# plt.savefig("MHS_Pred_VS_Test.png")
plt.show()

### PREDICTION PLOT MWHS2
colors = []

for idx, row in X_test_mwhs2.iterrows():
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

plt.figure(figsize=(8, 6))
scatter = plt.scatter(y_test_mwhs2, y_pred_mwhs2, c=colors, linewidths=1, alpha=0.5)
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
legend_handles = [Patch(facecolor=c, label=pred[i]) for i, c in enumerate(unique_colors)]
plt.legend(handles=legend_handles, loc='lower right', title='Predictors')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xticks()
plt.yticks()
plt.xlabel('True Betas')
plt.title('MWHS2: Predicted VS True Betas (MSE: {:.4f})'.format(mse_mwhs2))
# plt.savefig("MWHS2_Pred_VS_Test.png")
plt.show()

