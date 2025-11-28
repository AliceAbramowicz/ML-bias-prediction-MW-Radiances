import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.patches import Patch
import xskillscore

df_test = pd.read_csv("perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2021.csv")
X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")
"""
Rank histograms show the bias of the model:
    - if the forecast is accirate, the rank histogram is flat
"""

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

n_quantiles = y_pred.shape[1]
print("number of quantiles:", n_quantiles)

def rankhist(ens, obs):
    ens = np.array(ens, dtype='float64')
    obs = np.array(obs, dtype='float64')
    dim1 = ens.shape
    if len(dim1) == 1:
        ens = ens.reshape((1,dim1[0]))
    dim2 = obs.shape
    if len(dim2) == 0:
        obs = obs.reshape((1,1))
    elif len(dim2) == 1:
        obs = obs.reshape((dim2[0],1))

    nb_rows, nb_cols = ens.shape
    ranks = np.empty(nb_rows)
    ranks[:] =  np.nan

    for i in range(nb_rows):
        if ~np.isnan(obs[i]):
            if np.all(~np.isnan(ens[i,:])):
                ens_obs = np.append(ens[i,:], obs[i]) 
                #append true value to ensemble members and sort them
                ens_obs_sort = np.sort(ens_obs)
                # find index where the sorted array equals the true value
                idxs, = np.where(ens_obs_sort == obs[i])
                # store the index in rank array
                if len(idxs) > 1:
                    rand_idx = int(np.floor(np.random.rand(1) * len(idxs)))
                    ranks[i] = idxs[rand_idx]
                else:
                    ranks[i] = idxs[0]
    # remove NAs from ranks
    ranks_nonnan = ranks[~np.isnan(ranks)]
    # define bins  ranging from -0.5 to the number of columns plus 1.5
    bins = np.arange(0, nb_cols+1, 1)
    freq, _bin_edges = np.histogram(ranks_nonnan, bins=bins)
    # Normalize the frequencies to get relative frequencies
    rel_freq = freq/np.nansum(freq)

    return rel_freq, bins

### RANK HISTOGRAM ON FULL DATASET: 20 QUANTILES
rel_freq, bins = rankhist(y_pred, y_test)

plt.figure(figsize=(6, 6))
plt.bar(bins[1:len(bins)], rel_freq)
#sns.histplot(rel_freq, bins=np.linspace(0, 1, n_quantiles), kde=False)
plt.xlabel('Rank', fontsize=17)
plt.ylabel('Relative Frequency', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 1)
plt.grid(True)
plt.tight_layout()
#plt.savefig('rank_hist.png')
plt.show()
###############
ranks = np.zeros(len(y_test))
uniform_frequency = len(y_test) / (n_quantiles)
for i in range(len(y_test)):
    q_preds = y_pred[i, :] #quantile forecasts
    rank = np.sum(y_test[i] >= q_preds) / n_quantiles #the rank of the true value relative to the quantiles
    ranks[i] = rank

print("ranks:", ranks)

### RANK HISTOGRAM ON FULL DATASET: 20 QUANTILES
plt.figure(figsize=(6, 6))
sns.histplot(ranks, bins=np.linspace(0, 1, n_quantiles), kde=False)
plt.xlabel('Rank', fontsize=17)
plt.ylabel('Frequency', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 1)
plt.axhline(uniform_frequency, color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('rank_hist.png')
plt.show()

"""
### 10 QUANTILES
ranks = np.zeros(len(y_test))
horizontal_line = len(y_test) / 10
for i in range(len(y_test)):
    q_preds = y_pred[i, :]
    rank = np.sum(y_test[i] >= q_preds) / 10
    ranks[i] = rank

plt.figure(figsize=(6, 6))
sns.histplot(ranks, bins=np.linspace(0, 1, 11), kde=False)
plt.xlabel('Rank', fontsize=17)
plt.ylabel('Frequency', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axhline(uniform_frequency, color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()
"""
print("ranks:", ranks)

### SENSORS
colors = []
for idx, row in X_test.iterrows():
    if row['sensor_3'] == 1:
        colors.append('purple')
    elif row['sensor_15'] == 1:
        colors.append('darkturquoise')
    elif row['sensor_73'] == 1:
        colors.append('darkorange')
    else:
        colors.append('black')

rank_df = pd.DataFrame({
    'Rank': ranks,
    'Color': colors
})

sensor_labels = {
    'purple': 'AMSU-A',
    'darkturquoise': 'MHS',
    'darkorange': 'MWHS2'
}
plt.figure(figsize=(6, 6))
for color in rank_df['Color'].unique():
    subset = rank_df[rank_df['Color'] == color]
    plt.hist(subset['Rank'], bins=20, color=color, rwidth=0.05, histtype='step', linewidth=2.5, 
             align='mid', edgecolor=color, label=sensor_labels[color], stacked=True)
plt.xlabel('Rank', fontsize=17)
plt.ylabel('Frequency', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0,1)
plt.grid(True)
plt.legend(title='Sensors')
plt.tight_layout()
plt.savefig('ranked_hist_sensors.png')
plt.show()

### BIAS PREDICTORS
colors=[]
for idx, row in X_test.iterrows():
    if row['pred_id_0'] == 1:
        colors.append('purple')
    elif row['pred_id_1'] == 1:
        colors.append('blue')
    elif row['pred_id_2'] == 1:
        colors.append('green')
    elif row['pred_id_8'] == 1:
        colors.append('darkorange')
    elif row['pred_id_9'] == 1:
        colors.append('red')
    elif row['pred_id_10'] == 1:
        colors.append('gold')
    else:
        colors.append('black')

rank_df = pd.DataFrame({
    'Rank': ranks,
    'Color': colors
})

pred_labels = {
        'purple': 'constant',
        'blue': '1000-300hPa thickness',
        'green': '200-50hPa thickness',
        'darkorange': 'nadir view angle',
        'red': 'nadir view angle **2',
        'gold': 'nadir view angle **3'
}

plt.figure(figsize=(6, 6))
for color, label in pred_labels.items():
    subset = rank_df[rank_df['Color'] == color]
    plt.hist(subset['Rank'], bins=20, color=color, rwidth=0.05, histtype='step', linewidth=2.5,
             align='mid', edgecolor=color, label=label, stacked=True) 
plt.xlabel('Rank', fontsize=17)
plt.ylabel('Frequency', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 1)
plt.grid(True)
plt.legend(title='Bias predictors')
plt.tight_layout()
plt.savefig('ranked_hist_pred.png')
plt.show()

### CYCLES:
colors_cycles = []
cycles = df_test['cycle']
unique_cycles = cycles.unique()
colors_list = ['purple', 'blue', 'green', 'darkorange', 'red', 'gold']

for idx, row in df_test.iterrows():
    for i in range(len(unique_cycles)):
        if row['cycle'] == unique_cycles[i]:
            colors_cycles.append(colors_list[i])

rank_df = pd.DataFrame({
    'Rank': ranks,
    'Color': colors_cycles
})

cycle_labels = {
        'purple':'03:00 UTC',
        'blue':'06:00 UTC',
        'green':'09:00 UTC',
        'darkorange':'12:00 UTC',
        'red':'18:00 UTC',
        'gold':'21:00 UTC'
        }
plt.figure(figsize=(6, 6))
for color in rank_df['Color'].unique():
    subset = rank_df[rank_df['Color'] == color]
    plt.hist(subset['Rank'], bins=20, rwidth=0.05, color=color,  histtype='step', linewidth=2.5,
             align='mid', edgecolor=color, label=cycle_labels[color], stacked=True)
plt.xlabel('Rank', fontsize=17)
plt.ylabel('Frequency', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 1)
plt.grid(True)
plt.legend(title='Cycles')
plt.tight_layout()
plt.savefig('ranked_hist_cycle.png')
plt.show()
