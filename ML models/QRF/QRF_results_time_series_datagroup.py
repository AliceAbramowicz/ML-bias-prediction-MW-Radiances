import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib.patches import Patch
import properscoring as ps

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")
date_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2021.csv")

date = pd.to_datetime(date_test["time"], format='%Y%m%d_%H%M%S').dt.date
min_time = date.min()
max_time = date.max()

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
print("total CRPS:", crps)
def crps(y_true, ensemble):
    crps_per_obs = ps.crps_ensemble(y_true, ensemble)
    crps = np.mean(crps_per_obs)
    return crps

normalized_time = X_test["normalized_time"]
X_test["date"] = date
pred_ids = ['pred_id_0', 'pred_id_1', 'pred_id_2', 'pred_id_8', 'pred_id_9', 'pred_id_10']
pred_id_names = ['Constant', '1000-300hPa', '200-50hPa', 'nadir', 'nadir**2', 'nadir**3']

study_df = pd.DataFrame()
### PLOTS FOR METOP-B AMSU-A CHANNEL 8 CYCLE 9:
for pred_id, pred_id_name in zip(pred_ids, pred_id_names):
    mask = (X_test[pred_id] == 1) & (X_test['sat_3'] == 1) & (X_test['sensor_15'] == 1) & (X_test['channel'] == 4) & (X_test['cycle'] == 9)
    #mask = (X_test[pred_id] == 1) & (X_test['sat_523'] == 1) & (X_test['sensor_73'] == 1) & (X_test['channel'] == 14) & (X_test['cycle'] == 3)
    #mask = (X_test[pred_id] == 1) & (X_test['sat_3'] == 1) & (X_test['sensor_15'] == 1) & (X_test['channel'] == 5) & (X_test['cycle'] == 9)
    if mask.sum() > 0:
        y_test_datagroup = y_test[mask]
        y_test_datagroup = y_test[mask].values.reshape(-1)
        y_pred_datagroup = y_pred[mask]
        X_test_datagroup = X_test[mask]
        median = []
        quantile_5 = []
        quantile_95 = []
        date = X_test_datagroup["date"]
        normalized_time = X_test_datagroup["normalized_time"]
        quantile_5 = y_pred_datagroup[:, 1]
        quantile_95 = y_pred_datagroup[:, 19]
        median = y_pred_datagroup[:, 10]
        crps_datagroup = crps(y_test_datagroup, y_pred_datagroup)
        
        pred_study_df = X_test_datagroup[["date", "ndata","pred_id_0","pred_id_1","pred_id_2","pred_id_8","pred_id_9","pred_id_10"]]
        study_df = pd.concat([study_df, pred_study_df], axis=0)
        plt.figure(figsize=(6, 6))
        plt.scatter(date, median, c='blue', label='median prediction')
        plt.scatter(date, y_test_datagroup, c='lime', label='True Betas')
        plt.plot(date, median, c='blue')
        plt.plot(date, y_test_datagroup, c='lime')
        plt.fill_between(date, quantile_5, quantile_95, color='blue', alpha=0.3, label='90% CI')
        plt.xlabel('Date', fontsize=17)
        plt.ylabel('Predicted Value', fontsize=17)
        plt.xticks(fontsize=15, rotation=45)
        plt.yticks(fontsize=15)
        plt.legend(loc='upper right')
        plt.title(f'Metop-B MHS {pred_id} chan 5 cycle 9')
        plt.tight_layout()
        plt.savefig(f'QRF_time_series_MetopB_MHS_{pred_id}_chan5_9AM.png')
        plt.show()
        plt.close()


study_df.to_csv("datagroup_uncertainty.csv", index=False)

