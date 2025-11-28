import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.ensemble import AdaBoostRegressor
import random
import pickle
from sklearn.tree import DecisionTreeRegressor

np.random.seed(22)
random.seed(22)

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv")

# Ensure correct shape
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

base_tree = DecisionTreeRegressor(random_state=100)
model = AdaBoostRegressor(estimator=base_tree, random_state=100)

param_grid = {
        'estimator__max_depth': [4, 6],
        'n_estimators': [500],
        'learning_rate': [1.5, 1.75, 2], #[ 0.75, 1, 1.25, 1.5],
        'loss': ['square', 'exponential']
    }

# Initialize time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=2)
grid_result = grid_search.fit(X_train, y_train)

best_ab = grid_search.best_estimator_
best_params = grid_search.best_params_
cv_results = grid_search.cv_results_

print('best AB:', best_ab, 'best params:', best_params)

# Evaluate best model on test set
y_pred = best_ab.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)

best_result_dict = {'best_estimator': best_ab,
        'best_params': best_params,
        'cv_results': cv_results,
        'test_mse': mse_best,
        'r2': r2_best,
        'y_pred': y_pred
    }

pickle_file_best = 'AB_results_.pkl'
with open(pickle_file_best, 'wb') as f:
    pickle.dump(best_result_dict, f)

print("Best results saved in results_final.pkl")







