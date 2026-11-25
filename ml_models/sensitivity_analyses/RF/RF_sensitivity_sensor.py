import sys
import getopt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree
import random
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

np.random.seed(22)
random.seed(22)

X_train = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/X_train_stats.csv")
X_test = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/X_test_stats.csv")
y_train = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/y_train_stats.csv")
y_test = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiances/datasets/final_df/y_test_stats.csv")

def usage():
    print("Options:")
    print(" -s : sensor to remove from training (e.g. amsua, mhs, mwhs2)")

try:
    opts, args = getopt.getopt(sys.argv[1:], "s:h")
except getopt.GetoptError as err:
    print(str(err))
    usage()
    sys.exit(1)

for opt, arg in opts:
    if opt == '-h':
        usage()
        sys.exit(0)
    elif opt == '-s':
        sensor = arg

if sensor == 'DUMMY':
    print("No sensor specified, try -h for more info")
    sys.exit(1)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

print(X_train.columns)
print("Shape of X_train:", X_train.shape)

### Remove sensor from training:
sen_dict = {'amsua': 'sensor_3', 'mhs': 'sensor_15', 'mwhs2': 'sensor_73'}
print(f"removing {sen_dict[sensor]} from training")
mask = X_train[sen_dict[sensor]] != 1
X_train = X_train[mask]
y_train = y_train[mask]
print(f"Shape of X_train after removing {sensor}:", X_train.shape)

print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print(X_train)

param_grid = {
    'n_estimators': [500],
    'max_features': [8, 12, 16, 20, 24, 28],
    'min_samples_leaf': [4, 8, 12, 16],
}

model = RandomForestRegressor(random_state=100)

tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=2)
grid_result = grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)
feature_importances = best_rf.feature_importances_
cv_results = grid_search.cv_results_

best_result_dict = {'best_estimator': best_rf, 
        'feature_importances': feature_importances,
        'test_mse': mse, 
        'r2': r2,
        'cv_results': cv_results,
        'y_pred': y_pred_best
    }

pickle_file_RF = f'RF_sensitivity_{sensor}.pkl'

with open(pickle_file_RF, 'wb') as f:
    pickle.dump(best_result_dict, f)

print(f"Best results saved in RF_sensitivity_{sensor}.pkl")

