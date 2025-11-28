import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf

y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

pickle_file = "results_NN"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

pickle_file = "HISTORY_Bayesian.pkl"

with open(pickle_file, 'rb') as f:
    history = pickle.load(f)

model = tf.keras.models.load_model('NN_Bayesian_Q_T.keras')
print("model's architecture:")
model.summary()

y_pred_scaled = results_dict["y_pred_scaled"]
y_test_scaled = results_dict["y_test_scaled"]
y_test_original_scale = y_scaler.inverse_transform(y_test_scaled)
y_pred_original_scale = y_scaler.inverse_transform(y_pred_scaled)
print('type if y_pred:', type(y_pred_original_scale))
mse = mean_squared_error(y_test_original_scale, y_pred_original_scale) 
r2 = r2_score(y_test_original_scale, y_pred_original_scale)
print("mse:", round(mse,4), "r2", round(r2, 4))

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

### PREDICTION PLOT COLORED PER SENSOR:
color = []
for idx, row in X_test.iterrows():
    if row['sensor_3'] == 1:
        color.append('purple')
    elif row['sensor_15'] == 1:
        color.append('darkturquoise')
    elif row['sensor_73'] == 1:
        color.append('orange')
    else:
        color.append('black')

plt.figure(figsize=(6, 6))
plt.scatter(y_test_original_scale, y_pred_original_scale, alpha=0.5, c=colors)
min_val = min(min(y_test_original_scale), min(y_pred_original_scale))
max_val = max(max(y_test_original_scale), max(y_pred_original_scale))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlabel('True Betas', fontsize=15)
plt.ylabel('Predicted Betas', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.tight_layout()
#plt.title(f"NN (MSE={round(mse,4)}, R2={round(r2,4)})")
plt.savefig('NN_ytest_VS_ypred_colored_pred_exp.png')
plt.show()

### TRAINING HISTORY
plt.plot(figsize=(10, 10))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('ANN: loss evolution')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.gca().set_ylim(0, 0.3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("NN_best_loss_Bayesian_exp.png")
plt.show()

### PERFORMANCES PER BIAS PRED:
mask = X_test['pred_id_1'] == 1
y_test_pred1 = y_test_original_scale[mask]
y_pred_pred1 = y_pred_original_scale[mask]
X_test_pred1 = X_test[mask]
mse_pred1 = mean_squared_error(y_test_pred1, y_pred_pred1)
r2_pred1 = r2_score(y_test_pred1, y_pred_pred1)

mask = X_test['pred_id_2'] == 1
y_test_pred2 = y_test_original_scale[mask]
y_pred_pred2 = y_pred_original_scale[mask]
X_test_pred2 = X_test[mask]
mse_pred2 = mean_squared_error(y_test_pred2, y_pred_pred2)
print("total mse:", mse)
print("total r2:", r2)

# prediction plot per bias predictor
plt.figure(figsize=(6, 6))
scatter = plt.scatter(y_test_pred1, y_pred_pred1, linewidths=1, alpha=0.5)
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Betas')
plt.title('Tropospheric thickness 1000-300hPa (MSE: {:.4f})'.format(mse_pred1))
plt.show()

plt.figure(figsize=(6, 6))
scatter = plt.scatter(y_test_pred2, y_pred_pred2, linewidths=1, alpha=0.5)
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Betas')
plt.title('Tropospheric thickness 200-50hPa (MSE: {:.4f})'.format(mse_pred2))
plt.show()
