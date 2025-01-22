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

y_train = pd.read_csv("/perm/nld3863/create_dataset_all/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/create_dataset_all/y_test_stats.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats.csv")

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

pickle_file = "NN_results_BayesianOpt.pkl"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

pickle_file = "training_history_BayesianOpt.pkl"

with open(pickle_file, 'rb') as f:
    history = pickle.load(f)

model = tf.keras.models.load_model('NN_final_model_BayesianOpt.keras')
print("model's architecture:")
model.summary()

y_pred_scaled = results_dict["y_pred_scaled"]
y_test_scaled = results_dict["y_test_scaled"]
y_test_original_scale = y_scaler.inverse_transform(y_test_scaled)
y_pred_original_scale = y_scaler.inverse_transform(y_pred_scaled)
mse = mean_squared_error(y_test_original_scale, y_pred_original_scale) 
r2 = r2_score(y_test_original_scale, y_pred_original_scale)
print("mse:", mse, "r2", r2)

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
plt.savefig('NN_ytest_VS_ypred_colored_pred.png')
plt.show()

### TRAINING HISTORY
plt.plot(figsize=(10, 10))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('ANN: loss evolution')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.gca().set_ylim(0, 0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("NN_best_loss_Bayesian.png")
plt.show()

