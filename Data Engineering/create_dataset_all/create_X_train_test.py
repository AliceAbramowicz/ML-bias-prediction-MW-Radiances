import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df_train = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2023.csv")
df_test = pd.read_csv("/perm/nld3863/create_dataset_all/big_df_stats_2021.csv")

##### DIFFERENT PREDICTORS & MISSING VALUES:
"""
# 1) 5 predictors only:
df_train = df_train[df_train['npred'] == 5]
df_test = df_test[df_test['npred'] == 5]
df_train = df_train.drop(['predxcov_6'], axis=1)
df_test = df_test.drop(['predxcov_6'], axis=1)
print("df train 5 preds:", df_train.shape)

# 2) all predictors, remove columns with NAs
df_train = df_train.drop(['predxcov_5','predxcov_6'], axis=1)
df_test = df_test.drop(['predxcov_5','predxcov_6'], axis=1)
"""
# 3) all predictors, impute NAs with 0s
df_train.replace('', float('nan'), inplace=True)
df_test.replace('', float('nan'), inplace=True)
df_train['predxcov_5'].fillna(0, inplace=True)
df_train['predxcov_6'].fillna(0, inplace=True)
df_test['predxcov_5'].fillna(0, inplace=True)
df_test['predxcov_6'].fillna(0, inplace=True)

#####

y_train = df_train["param"]
y_test = df_test["param"]

y_train_to_sc = y_train.values.reshape(-1, 1)
y_test_to_sc = y_test.values.reshape(-1, 1)

scaler_y = StandardScaler()

y_train_scaled = scaler_y.fit_transform(y_train_to_sc) 
# fit scaler on training set only to prevent info leakage from test set
y_test_scaled = scaler_y.transform(y_test_to_sc)

X_train = df_train.drop(columns=["param", "param0", "time"])
X_test = df_test.drop(columns=["param", "param0", "time"])

categories = ["pred_id", "sat", "sensor"]
numeric_columns = [col for col in X_train.columns if col not in categories and col != 'cycle']
X_train['cycle_cos'] = np.cos(X_train['cycle'] * 2 * np.pi / 24)
X_train['cycle_cos'] = X_train['cycle_cos'].round(6)
X_test['cycle_cos'] = np.cos(X_test['cycle'] * 2 * np.pi / 24)
X_test['cycle_cos'] = X_test['cycle_cos'].round(6)

#X_train = X_train.drop(columns=["cycle"])
#X_test = X_test.drop(columns=["cycle"])

X_train = pd.get_dummies(X_train, columns=categories)
X_test = pd.get_dummies(X_test, columns=categories)

X_train.to_csv("X_train_stats.csv", index=False)
X_test.to_csv("X_test_stats.csv", index=False)
y_test.to_csv("y_test_stats.csv", index=False)
y_train.to_csv("y_train_stats.csv", index=False)

X_train_scaled = X_train
X_test_scaled = X_test

scaler_X = StandardScaler()
X_train_scaled[numeric_columns] = scaler_X.fit_transform(X_train_scaled[numeric_columns])
X_test_scaled[numeric_columns] = scaler_X.transform(X_test_scaled[numeric_columns])

print("Training Feature Names:", X_train.columns.tolist())
print("Test Feature Names:", X_test.columns.tolist())
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

y_train_scaled = pd.Series(y_train_scaled.flatten(), name = "param")
y_test_scaled = pd.Series(y_test_scaled.flatten(), name = "param")

X_train_scaled.to_csv("X_train_stats_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_stats_scaled.csv", index=False)
y_test_scaled.to_csv("y_test_stats_scaled.csv", index=False)
y_train_scaled.to_csv("y_train_stats_scaled.csv", index=False)

