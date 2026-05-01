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
    print(" -m <model> : model (RF, AB, XGB, NN)")
    print(" -s <True/False> : are your results scaled?")
    print(" -f <True/False> : show feature importance?")
    print(" -S <True/False> : plot results colored per satellite?")
    print(" -h : help")

# defaults
scaled = False
show_feature_imp = False
show_sat_plot = False
show_biaspred_plot = True
show_sensor_plot = True
model = None

def str_to_bool(x):
    return str(x).lower() in ("true", "1", "yes", "y")

try:
    opts, args = getopt.getopt(sys.argv[1:], "m:s:f:S:h")
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
        scaled = str_to_bool(arg)
    elif opt == '-f':
        show_feature_imp = str_to_bool(arg)
    elif opt == '-S':
        show_sat_plot = str_to_bool(arg)

if model == "DUMMY":
    print("No model specified, try -h for more info")
    sys.exit(1)

pickle_file = f"{model}/{model}_results.pkl"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

if scaled == True: # NN
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    y_pred_scaled = results_dict["y_pred_scaled"]
    y_test_scaled = results_dict["y_test_scaled"]
    y_test = y_scaler.inverse_transform(y_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)

elif scaled == False: # XGB, AB, RF
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    # best_model = results_dict["best_estimator"]
    # print(best_model)
    mse = results_dict["mse"]
    r2 = results_dict["r2"]
    print('r2:', r2)
    print('mse:', mse)
    y_pred = results_dict["y_pred"]

# ====================================
### FEATURE IMPORTANCE PLOT:
# ====================================
if show_feature_imp:
    feature_imp = results_dict["feature_importances"]
    idx = feature_imp.argsort()
    top_features_idx = idx[-25:]
    plt.barh(X_train.columns[top_features_idx], feature_imp[top_features_idx])
    plt.xlabel("feature importance")
    plt.title("Random Forest: Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{model}/{model}_feature_importance.png')
    plt.close()

# ====================================
### MAKE PLOTS PER SELECTED CATEGORIES
# ====================================
# bias preds
if show_biaspred_plot:
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
    plt.xlim(-2.6, 1.8)
    plt.ylim(-2.6, 1.8)
    plt.xlabel('True Betas', fontsize=15)
    plt.ylabel('Predicted Betas', fontsize=15)
    plt.xticks(np.arange(-2.5, 1.8, 0.5), fontsize=13)
    plt.margins(0.01)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'{model}/{model}_preds_ytest_VS_ypred.png')
    plt.show()       

# Sensors:
if show_sensor_plot:
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
    plt.savefig(f'{model}/{model}_sensors_ytest_VS_ypred.png')
    plt.show()  

# Satellites:
if show_sat_plot:
    colors = []

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
    plt.savefig(f'{model}/{model}_sats_ytest_VS_ypred.png')
    plt.show()  

# With hPa column
channel_values = X_test["peak_hPa"].values
print("min & max channels:", channel_values.min(), channel_values.max())
plt.figure(figsize=(10, 10))
scatter = plt.scatter(y_test, y_pred, c=channel_values, cmap="RdBu_r", linewidths=1, alpha=0.7)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlabel('True Betas')
plt.ylabel('Predicted Betas')
cbar = plt.colorbar(scatter)
cbar.set_label('hPa')
plt.savefig(f'{model}/{model}_channels_ytest_VS_ypred.png')
plt.show()

