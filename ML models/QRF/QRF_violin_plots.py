import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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

df_test['quantile_5'] = quantile_5
df_test['quantile_95'] = quantile_95
df_test['median'] = median
df_test['CI_width'] = quantile_95 - quantile_5

### SENSORS:
sensor_map = {3: 'AMSU-A', 15: 'MHS', 73: 'MWHS2'}

sensor_palette = {
    'AMSU-A': 'purple',
    'MHS': 'darkturquoise',
    'MWHS2': 'orange'
}

df_test['sensor'] = df_test['sensor'].map(sensor_map)
df_test['sensor'] = pd.Categorical(df_test['sensor'], categories=sensor_palette.keys(), ordered=True)
plt.figure(figsize=(5,5))
sns.violinplot(data=df_test, x='sensor', y='CI_width', points='all', palette=sensor_palette)
plt.ylabel('90% CI Width', fontsize=14)
plt.xlabel('Sensors', fontsize=14)
plt.xticks(ticks=range(len(sensor_palette)), labels=sensor_palette.keys(), rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title('PDFs of 90% CI width per sensor')
plt.tight_layout()
plt.savefig('violin_plot_90CI_sensor.png')
plt.show()

### BIAS PREDICTORS:
pred_map = {
        0: 'constant',
        1: '1000-300hPa thickness',
        2: '200-50hPa thickness',
        8: 'nadir view angle',
        9: 'nadir view angle **2',
        10: 'nadir view angle **3'
}
pred_palette = {
    'constant': 'purple',
    '1000-300hPa thickness': 'blue',
    '200-50hPa thickness': 'green',
    'nadir view angle': 'orange',
    'nadir view angle **2': 'red',
    'nadir view angle **3': 'yellow'
}
df_test['pred_id'] = df_test['pred_id'].map(pred_map)
df_test['pred_id'] = pd.Categorical(df_test['pred_id'], categories=pred_palette.keys(), ordered=True)

plt.figure(figsize=(5, 5))
sns.violinplot(data=df_test, x='pred_id', y='CI_width', points='all', palette=pred_palette)
plt.ylabel('90% CI Width', fontsize=14)
plt.xlabel('Bias Predictors', fontsize=14)
plt.xticks(ticks=range(len(pred_palette)), labels=pred_palette.keys(), rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title('PDFs of 90% CI width per predictor')
plt.tight_layout()
plt.savefig('violin_plot_90CI_pred.png')
plt.show()

### CHANNELS:
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_test, x='channel', y='CI_width', points='all')
plt.ylabel('90% CI Width')
plt.xlabel('Channel')
plt.title('PDFs of 90% CI width per channel')
plt.show()

### CYCLES:
cycle_map = {
        3: '03:00 UTC',
        6: '06:00 UTC',
        9: '09:00 UTC',
        12: '12:00 UTC',
        18: '18:00 UTC',
        21: '21:00 UTC'
        }

color_map = {
        '03:00 UTC': 'purple',
        '06:00 UTC': 'blue',
        '09:00 UTC': 'green',
        '12:00 UTC': 'orange',
        '18:00 UTC': 'red',
        '21:00 UTC': 'yellow'
        }
df_test['cycle'] = df_test['cycle'].map(cycle_map)
df_test['cycle'] = pd.Categorical(df_test['cycle'], categories=color_map.keys(), ordered=True)
plt.figure(figsize=(5, 5))
sns.violinplot(data=df_test, x='cycle', y='CI_width',  points='all', palette=color_map)
plt.ylabel('90% CI Width')
plt.xticks(ticks=range(len(cycle_map)), labels=color_map.keys(), rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('90% CI Width', fontsize=14)
plt.xlabel('Cycles', fontsize=14)
plt.title('PDFs of 90% CI width per cycle')
plt.tight_layout()
plt.savefig('violin_plot_90CI_cycle.png')
plt.show()

### SATELLITES:
sat_map = {
        3: 'Metop B',
        5: 'Metop C',
        209: 'NOAA 18',
        223: 'NOAA 19',
        523: 'FY-3D'}
color_map = {
        'Metop B': 'grey',
        'Metop C': 'blue',
        'NOAA 18': 'green',
        'NOAA 19': 'purple',
        'FY-3D': 'orange'}

df_test['sat'] = df_test['sat'].map(sat_map)
df_test['sat'] = pd.Categorical(df_test['sat'], categories=color_map.keys(), ordered=True)

plt.figure(figsize=(5, 5))
sns.violinplot(data=df_test, x='sat', y='CI_width',  points='all', palette=color_map)
plt.ylabel('90% CI Width')
plt.xticks(ticks=range(len(sat_map)), labels=color_map.keys(), rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('90% CI Width', fontsize=16)
plt.xlabel('Satellites', fontsize=16)
plt.title('PDFs of 90% CI width per satellite')
plt.tight_layout()
plt.savefig('violin_plot_90CI_sat.png')
plt.show()
