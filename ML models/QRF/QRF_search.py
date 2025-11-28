from skranger.ensemble import RangerForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import GridSearchCV
import pickle
import random
import properscoring as ps
import numpy as np

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")

np.random.seed(22)
random.seed(22)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
        'n_estimators': [250, 500],
        'mtry': [10, 20, 25, 30],
        'min_node_size': [0, 1, 2, 4, 6]
        }

quantiles = np.linspace(0, 1, num=21)

def crps(y_true, ensemble):
    crps_per_obs = ps.crps_ensemble(y_true, ensemble)
    crps = np.mean(crps_per_obs)
    return crps

crps_scorer = make_scorer(crps, greater_is_better=False)

qrf = RangerForestRegressor(quantiles=True)
grid_search = GridSearchCV(estimator=qrf, param_grid=param_grid, scoring=crps_scorer, verbose=True)
grid_search.fit(X_train, y_train)

best_qrf = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred_best = best_qrf.predict_quantiles(X_test, quantiles=quantiles)
cv_results = grid_search.cv_results_
crps_score = crps(y_test, y_pred_best)

best_result_dict = {'best_estimator': best_qrf,
        'best_params': best_params,
        'y_pred': y_pred_best,
        'cv_results': cv_results,
        'crps': crps_score
    }

pickle_file = 'QRF_results.pkl'

with open(pickle_file, 'wb') as f:
    pickle.dump(best_result_dict, f)

print(f"Results saved in '{pickle_file}'")










