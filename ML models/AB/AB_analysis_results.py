import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Patch

df_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2023.csv")
df_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2021.csv")
X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv('/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv')
y_test = pd.read_csv('/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv')

y_test = np.ravel(y_test)

pickle_file = "AB_FINAL.pkl"

with open(pickle_file, 'rb') as f:
    results_dict = pickle.load(f)

with open(pickle_file, 'rb') as f:
    GS = pickle.load(f)

print("GS:", GS)

best_AB = results_dict["best_estimator"]
cv_results = results_dict["cv_results"]
mse = results_dict["test_mse"]
r2 = results_dict["r2"]
y_pred = results_dict["y_pred"]
feature_imp = best_AB.feature_importances_
print("best Adaboost model:", best_AB)
print("r2:", r2)
print("mse:", mse)

### FEATURE IMPORTANCE PLOT:
idx = feature_imp.argsort()
top_features_idx = idx[-18:]
plt.figure(figsize=(6,4))
plt.barh(X_train.columns[top_features_idx], feature_imp[top_features_idx])
plt.xlabel("feature importance")
plt.tight_layout()
plt.savefig('Feature_importance.png')
plt.show()
plt.close()

# PREDICTION PLOT COLORED BY PREDICTORS
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

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, c=colors)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel('True Betas', fontsize=15)
plt.ylabel('Predicted Betas', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig("AB_ytest_VS_ypred_colored_preds.png")
plt.show()
