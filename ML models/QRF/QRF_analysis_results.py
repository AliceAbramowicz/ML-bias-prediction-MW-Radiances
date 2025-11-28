import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")

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

print("cv_results: mean_test_score", cv_results["mean_test_score"])
print("best QRF:", best_QRF)
print("crps:", crps)
print("y pred:", y_pred, y_pred.shape)
print("quantiles:", quantiles)
print("quantile 5:", quantile_5)
print("quantile 95:", quantile_95)
print("median:", median)
"""
MEDIAN = best_QRF.predict_quantiles(X_test, quantiles=[0.5])
QUANTILE_95 = best_QRF.predict_quantiles(X_test, quantiles=[0.95])

print("QUANTILE_95:", QUANTILE_95)
print("MEDIAN:", MEDIAN)
"""
mtry_unique = np.unique(cv_results["param_mtry"])
min_node_size_unique = np.unique(cv_results["param_min_node_size"])
max_est_unique = np.unique(cv_results["param_n_estimators"])
print("Unique mtry values:", mtry_unique)
print("Unique min node size:", min_node_size_unique)
print("unique max estimators:", max_est_unique)

colors = []
"""
##### PREDICTORS:
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

colors = []
##### SATELLITES:
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
"""
colors = []
######SENSORS:
for idx, row in X_test.iterrows():
    if row['sensor_3'] == 1:
        colors.append('purple')
    elif row['sensor_15'] == 1:
        colors.append('blue')
    elif row['sensor_73'] == 1:
        colors.append('orange')
    else:
        colors.append('black')

### QUANTILES PLOT FOR A GIVEN SENSOR:
# Keep only MHS:
mask = X_test['sensor_15'] == 1
y_test = y_test[mask]
median = median[mask]
quantile_5 = quantile_5[mask]
quantile_95 = quantile_95[mask]

plt.figure(figsize=(10, 6))
X = np.linspace(y_test.min(), y_test.max(), len(y_test))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='perfect fit')
plt.scatter(y_test, quantile_5, label='Predicted 5th percentile', color="orange", alpha=0.4)
plt.scatter(y_test, quantile_95, label='Predicted 95th percentile', color="green", alpha=0.4)
plt.scatter(y_test, median, label='Predicted median', color="black")
plt.title('MHS: Quantile Random Forest Predictions')
plt.legend()
plt.show()

### QUANTILE PLOT ALL DATA:
plt.figure(figsize=(8, 6))
X = np.linspace(y_test.min(), y_test.max(), len(y_test))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='1:1 line')
plt.scatter(y_test, quantile_5, label='Predicted 5th percentile', color="orange", alpha=0.4)
plt.scatter(y_test, quantile_95, label='Predicted 95th percentile', color="green", alpha=0.4)
plt.scatter(y_test, median, label='Predicted median', color="black")
plt.title('Quantile Random Forest Predictions')
plt.legend()
plt.show()

# PREDICTIONS plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(y_test, median, alpha=0.6, label='Predicted Median')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='1:1 line')
X = np.linspace(y_test.min(), y_test.max(), len(y_test))
plt.fill_between(X, X-abs(quantile_5), X+abs(quantile_95), color='red', alpha=0.2, label='5th-95th Percentile Interval')
plt.title('Quantile Random Forest Predictions')
plt.legend()
plt.show()

# PREDICTIONS plot colored by sensor
"""
plt.figure(figsize=(8, 6))
scatter = plt.scatter(y_test, median, alpha=0.6, label='Predicted Median', c = colors)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='1:1 line')
X = np.linspace(y_test.min(), y_test.max(), len(y_test))
plt.fill_between(X, X-abs(quantile_5), X+abs(quantile_95), color='red', alpha=0.2, label='5th-95th Percentile Interval')
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
unique_colors = ["purple", "blue", "orange"]
legend_handles = [Patch(facecolor=c, label=sensor_labels[i]) for i, c in enumerate(unique_colors)]
plt.legend(handles=legend_handles, loc='upper right', title='Sensors')
plt.title('Quantile Random Forest Predictions')
plt.legend()
plt.show()
"""
### UNCERTAINTY PLOT: 90% CI (COLORED IN CATEGORIES OF YOUR CHOICE)
interval = quantile_95 - quantile_5
sort_ind = np.argsort(interval)
print("sort ind", sort_ind)
y_true_all = []
y_true_all = np.concatenate((y_true_all, y_test))
y_true_all_sorted = y_true_all[sort_ind]
colors_sorted = [colors[i] for i in sort_ind]
sensors = X_test["sensor_3"]
sensors_sorted = [sensors[i] for i in sort_ind]
#print("colors sorted:", colors_sorted)
#print("sensor 3 sorted:", sensors_sorted)
#print("type of colors sorted:", type(colors_sorted))
quantile_5_sorted = quantile_5[sort_ind]
quantile_95_sorted = quantile_95[sort_ind]
mean = (quantile_5_sorted + quantile_95_sorted)/2
# center the prediction around 0
y_true_all_centered = y_true_all_sorted - mean
upper_centered = quantile_95_sorted - mean
lower_centered = quantile_5_sorted - mean

length_y = len(y_true_all_centered)
samples_order = np.arange(1,  length_y+1)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(samples_order, y_true_all_centered, c=colors_sorted, linewidths=1, alpha=0.6)
plt.fill_between(np.arange(len(upper_centered)), lower_centered, upper_centered, alpha=0.3, color="r", label="Prediction 90th percentile")
sensor_labels = {
    0: 'AMSU-A',
    1: 'MHS',
    2: 'MWHS2'
}
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
#unique_colors = ["purple", "blue", "green", "orange", "red", "yellow"]
legend_handles = [Patch(facecolor=c, label=sensor_labels[i]) for i, c in enumerate(unique_colors)]
plt.legend(handles=legend_handles, loc='upper left', title='Sensors')
plt.xlabel("Ordered & centered samples")
plt.ylabel("True Betas and predicted 90% CI")
plt.title(f"Predicted 90% confidence interval (CRPS: {round(crps, 4)})")
plt.xlim([0, len(y_true_all)])
plt.show()
