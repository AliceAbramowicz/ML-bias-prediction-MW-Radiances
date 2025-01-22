import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.ensemble import AdaBoostRegressor
import random
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

np.random.seed(22)
random.seed(22)

X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/create_dataset_all/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/create_dataset_all/y_test_stats.csv")

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
        'estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=10)],
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.5, 1, 1.5, 2],
    'loss': ['linear', 'square', 'exponential']
    }

model = AdaBoostRegressor(random_state=100)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)
grid_result = grid_search.fit(X_train, y_train)

best_ab = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred_best = best_ab.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
cv_results = grid_search.cv_results_

best_result_dict = {'best_estimator': best_ab,
        'best_params': best_params,
        'cv_results': cv_results,
        'test_mse': mse_best,
        'r2': r2_best,
        'y_pred': y_pred_best
    }

pickle_file_best = 'AB_best_result_FINAL.pkl'
with open(pickle_file_best, 'wb') as f:
    pickle.dump(best_result_dict, f)

print("Best results saved in AB_best_result_FINAL.pkl")








