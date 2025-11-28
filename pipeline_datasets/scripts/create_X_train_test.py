import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2023.csv")
df_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_2021.csv")

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
df_train.loc[:, df_train.columns.str.startswith('predxcov_')] = (df_train.loc[:, df_train.columns.str.startswith('predxcov_')].fillna(0))

df_test.loc[:, df_test.columns.str.startswith('predxcov_')] = (df_test.loc[:, df_test.columns.str.startswith('predxcov_')].fillna(0))

# Drop rows with NaN values in all columns except predxcov (because those just have fewer bias predictors)
df_train = df_train.dropna(axis=0)
df_test = df_test.dropna(axis=0)

#####

y_train = df_train["param"]
y_test = df_test["param"]
print('length of y_test', len(y_test))
print('shape of df_test:', df_test.shape)
y_train_to_sc = y_train.values.reshape(-1, 1)
y_test_to_sc = y_test.values.reshape(-1, 1)

scaler_y = StandardScaler()

y_train_scaled = scaler_y.fit_transform(y_train_to_sc) 
# fit scaler on training set only to prevent info leakage from test set
y_test_scaled = scaler_y.transform(y_test_to_sc)

X_train = df_train.drop(columns=["param", "param0", "time"])
X_test = df_test.drop(columns=["param", "param0", "time"])

categories = ["pred_id", "sat", "sensor", "channel"]

X_train['cycle_cos'] = np.cos(X_train['cycle'] * 2 * np.pi / 24)
X_train['cycle_cos'] = X_train['cycle_cos'].round(6)
X_test['cycle_cos'] = np.cos(X_test['cycle'] * 2 * np.pi / 24)
X_test['cycle_cos'] = X_test['cycle_cos'].round(6)

numeric_columns = [col for col in X_train.columns if col not in categories]
#X_train = X_train.drop(columns=["cycle"])
#X_test = X_test.drop(columns=["cycle"])

X_train = pd.get_dummies(X_train, columns=categories)
X_test = pd.get_dummies(X_test, columns=categories)

X_train.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats.csv", index=False)
X_test.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats.csv", index=False)
y_test.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats.csv", index=False)
y_train.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats.csv", index=False)

X_train_scaled = X_train
X_test_scaled = X_test

X_train_scaled = X_train_scaled.astype(float)
X_test_scaled  = X_test_scaled.astype(float)

scaler_X = StandardScaler()
X_train_scaled[numeric_columns] = scaler_X.fit_transform(X_train_scaled[numeric_columns])
X_test_scaled[numeric_columns] = scaler_X.transform(X_test_scaled[numeric_columns])

print("Training Feature Names:", X_train.columns.tolist())
print("Test Feature Names:", X_test.columns.tolist())
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

y_train_scaled = pd.Series(y_train_scaled.flatten(), name = "param")
y_test_scaled = pd.Series(y_test_scaled.flatten(), name = "param")

X_train_scaled.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats_scaled.csv", index=False)
X_test_scaled.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats_scaled.csv", index=False)
y_test_scaled.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats_scaled.csv", index=False)
y_train_scaled.to_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats_scaled.csv", index=False)

