import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch

df_test = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2021.csv")
X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats.csv")
y_test = df_test["param"]

pickle_file = "QRF_results.pkl"
with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

best_QRF = results_dict["best_estimator"]
cv_results = results_dict["cv_results"]
y_pred = results_dict["y_pred"]
crps = results_dict["crps"]
quantiles = np.linspace(0, 1, num=21)
quantile_5 = y_pred[:, 1]
quantile_95 = y_pred[:, 19]
median = y_pred[:, 10]
print(cv_results)
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

### UNCERTAINTY PLOT: 90% CI (COLORED IN CATEGORIES OF YOUR CHOICE)
interval = quantile_95 - quantile_5
sort_ind = np.argsort(interval)
y_true_all = []
y_true_all = np.concatenate((y_true_all, y_test))
y_true_all_sorted = y_true_all[sort_ind]
colors_sorted = [colors[i] for i in sort_ind]
quantile_5_sorted = quantile_5[sort_ind]
quantile_95_sorted = quantile_95[sort_ind]
mean = (quantile_5_sorted + quantile_95_sorted)/2
# center the prediction around 0
y_true_all_centered = y_true_all_sorted - mean
upper_centered = quantile_95_sorted - mean
lower_centered = quantile_5_sorted - mean

length_y = len(y_true_all_centered)
samples_order = np.arange(1,  length_y+1)

plt.figure(figsize=(6,6))
scatter = plt.scatter(samples_order, y_true_all_centered, c=colors_sorted, linewidths=1, alpha=0.6)
plt.fill_between(np.arange(len(upper_centered)), lower_centered, upper_centered, alpha=0.3, color="blue", label="Prediction 90th percentile")
unique_colors = ["purple", "darkturquoise", "orange"]
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
plt.xlabel("Ordered & centered samples", fontsize=14)
plt.ylabel("True Betas and predicted 90% CI", fontsize=14)
plt.xlim([0, len(y_true_all)])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig('QRF_uncertainty_sensors.png')
plt.show()

# bias predictors:
colors=[]
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
### UNCERTAINTY PLOT: 90% CI (COLORED IN BIAS PREDICTORS)
interval = quantile_95 - quantile_5
sort_ind = np.argsort(interval)
y_true_all = []
y_true_all = np.concatenate((y_true_all, y_test))
y_true_all_sorted = y_true_all[sort_ind]
colors_sorted = [colors[i] for i in sort_ind]
quantile_5_sorted = quantile_5[sort_ind]
quantile_95_sorted = quantile_95[sort_ind]
mean = (quantile_5_sorted + quantile_95_sorted)/2
# center the prediction around 0
y_true_all_centered = y_true_all_sorted - mean
upper_centered = quantile_95_sorted - mean
lower_centered = quantile_5_sorted - mean

length_y = len(y_true_all_centered)
samples_order = np.arange(1,  length_y+1)
plt.figure(figsize=(6,6))
scatter = plt.scatter(samples_order, y_true_all_centered, c=colors_sorted, linewidths=1, alpha=0.6)
plt.fill_between(np.arange(len(upper_centered)), lower_centered, upper_centered, alpha=0.3, color="blue", label="Prediction 90th percentile")
unique_colors = ["purple", "blue", "green", "orange", "red", "yellow"]
pred = {
        0: 'constant',
        1: '1000-300hPa thickness',
        2: '200-50hPa thickness',
        3: 'nadir view angle',
        4: 'nadir view angle **2',
        5: 'nadir view angle **3'
}
plt.xlabel("Ordered & centered samples", fontsize=14)
plt.ylabel("True Betas and predicted 90% CI", fontsize=14)
plt.xlim([0, len(y_true_all)])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig('QRF_uncertainty_predictors.png')
plt.show()

# cycles
colors_cycles = []
cycles = df_test['cycle']
unique_cycles = cycles.unique()
colors_list = ['purple', 'blue', 'green', 'orange', 'red', 'yellow']

for idx, row in df_test.iterrows():
    for i in range(len(unique_cycles)):
        if row['cycle'] == unique_cycles[i]:
            colors_cycles.append(colors_list[i])

colors_sorted = [colors_cycles[i] for i in sort_ind]
quantile_5_sorted = quantile_5[sort_ind]
quantile_95_sorted = quantile_95[sort_ind]
mean = (quantile_5_sorted + quantile_95_sorted)/2
y_true_all_centered = y_true_all_sorted - mean
upper_centered = quantile_95_sorted - mean
lower_centered = quantile_5_sorted - mean

samples_order = np.arange(1,  length_y+1)
plt.figure(figsize=(6,6))
scatter = plt.scatter(samples_order, y_true_all_centered, c=colors_sorted, linewidths=1, alpha=0.6)
plt.fill_between(np.arange(len(upper_centered)), lower_centered, upper_centered, alpha=0.3, color="blue", label="Prediction 90th percentile")
unique_colors = ["purple", "blue", "green", "orange", "red", "yellow"]
cycle_labels = {
        0: unique_cycles[0],
        1: unique_cycles[1],
        2: unique_cycles[2],
        3: unique_cycles[3],
        4: unique_cycles[4],
        5: unique_cycles[5]
}
legend_handles = [Patch(facecolor=c, label=cycle_labels[i]) for i, c in enumerate(unique_colors)]
plt.xlabel("Ordered & centered samples", fontsize=14)
plt.ylabel("True Betas and predicted 90% CI", fontsize=14)
plt.xlim([0, len(y_true_all)])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig('QRF_uncertainty_cycles.png')
plt.show()


