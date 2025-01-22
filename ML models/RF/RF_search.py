import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from permetrics.regression import RegressionMetric
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn import tree
from sklearn.tree import plot_tree
import random
import pickle
from sklearn.model_selection import GridSearchCV

np.random.seed(22)
random.seed(22)

X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/create_dataset_all/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/create_dataset_all/y_test_stats.csv")

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

print(X_train.columns)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print(X_train)

param_grid = {
    'n_estimators': [250, 500],
    'max_features': [12, 16, 20, 24, 28, 32],
    'min_samples_leaf': [1, 2, 4, 8, 12]
}

model = RandomForestRegressor(random_state=100)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)
grid_result = grid_search.fit(X_train, y_train)

with open('RF_GS.pkl', 'wb') as file:
    pickle.dump(grid_search, file)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
feature_importances = best_rf.feature_importances_
cv_results = grid_search.cv_results_

best_result_dict = {'best_estimator': best_rf, 
        'cv_results': cv_results,
        'feature_importances': feature_importances,
        'mse': mse_best, 
        'r2': r2_best,
        'y_pred': y_pred_best
    }

pickle_file_best = 'RF_best_result.pkl'

with open(pickle_file_best, 'wb') as f:
    pickle.dump(best_result_dict, f)

print("Best results saved in RF_best_result.pkl")

