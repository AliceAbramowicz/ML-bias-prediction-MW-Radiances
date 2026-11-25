import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import random
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

tf.random.set_seed(100)
np.random.seed(100)
random.seed(100)
keras.utils.set_random_seed(100) # repoducible weight shuffling

n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))
tf.config.threading.set_intra_op_parallelism_threads(n_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

suffix = "_nwpcy43_simple" # if you run several models, save them with different suffix names

X_train = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/X_train_stats_scaled.csv")
X_val = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/X_val_stats_scaled.csv")
X_test = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/X_test_stats_scaled.csv")
y_train = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/y_train_stats_scaled.csv")
y_val = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/y_val_stats_scaled.csv")
y_test = pd.read_csv("/perm/nld3863/ML-bias-prediction-MW-Radiance/datasets/final_df/y_test_stats_scaled.csv")

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

class NNHyperModel(kt.HyperModel):

    def build(self, hp):
        model = keras.Sequential()

        use_l2 = hp.Boolean('use_l2_regularization')
        if use_l2:
            l2_reg = hp.Float('l2_reg', 1e-4, 1e-2, sampling='log')
            kernel_regularizer = keras.regularizers.l2(l2_reg)
        else:
            kernel_regularizer = None
        
        kernel_init = hp.Choice('kernel_initializer', ['he_normal', 'he_uniform'])
        units1 = hp.Int('units1', min_value=8, max_value=80, step=8)

        model.add(keras.Input(shape=X_train.shape[1:]))

        bn1 = hp.Boolean('batch_norm1')
        model.add(layers.Dense(units=units1,
                               kernel_regularizer=kernel_regularizer,
                               kernel_initializer=kernel_init,
                               use_bias=not bn1))
        if bn1:
            model.add(layers.BatchNormalization())  # recenter and rescale the batch
        model.add(layers.Activation("relu"))

        if hp.Boolean('dropout1'):
            model.add(layers.Dropout(rate=0.25))
        
        for i in range(2, 4):
            if hp.Boolean(f'add_layer_{i}'):
                bn_i = hp.Boolean(f'batch_norm{i}')
                model.add(layers.Dense(units=hp.Int(f'units{i}', min_value=8, max_value=96, step=8),
                                       kernel_regularizer=kernel_regularizer,
                                       kernel_initializer=kernel_init,
                                       use_bias=not bn_i))
                if bn_i:
                    model.add(layers.BatchNormalization())
                model.add(layers.Activation("relu"))
                if hp.Boolean(f'dropout{i}'):
                    model.add(layers.Dropout(rate=0.25))

        model.add(layers.Dense(1))

        learning_rate = hp.Float('adam_learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss="mean_squared_error")
        return model
    # search for optimal batch size:
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[64, 128, 256, 512]),
            shuffle=True,
            **kwargs
        )

################### Keras Tuner library for hyperparameter optimization ###################
tuner = kt.BayesianOptimization(NNHyperModel(),
                     objective='val_loss',
                     max_trials=50,
                     num_initial_points=None, # without this the GP never gets fitted
                     alpha=0.0001,
                     beta=2.6,
                     seed=100,
                     tune_new_entries=True,
                     overwrite=True, #  False if want to resume
                     directory='kt_runs',
                     project_name=f'bayesian_search_NN{suffix}')

stop_early = EarlyStopping(
    monitor='val_loss',
    patience=5,
)

################### IF YOU NEED TO SEARCH FOR HYPERPARAM ###################
# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_batch_size = best_hps.get('batch_size')
print(best_hps.values)
print('batch size', best_batch_size)

stop_early_ = EarlyStopping(
    monitor='val_loss',
    patience=10, #more patience for training than hyperparam search
    restore_best_weights=True)

# ################## IF YOU NEED TO TRAIN YOUR MODEL ###################
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=1000, batch_size=best_batch_size, validation_data=(X_val, y_val), callbacks=[stop_early_])
with open(f'HISTORY_Bayesian{suffix}.pkl', 'wb') as f:
   pickle.dump(history.history, f)
model.save(f"NN_Bayesian{suffix}.keras")


################### IF MODEL ALREADY TRAINED, LOAD IT ###################
# model = tf.keras.models.load_model(f'NN_Bayesian{suffix}.keras')

print("model's architecture:")
model.summary()

y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)
test_mse = mean_squared_error(y_test, y_pred)
val_mse = mean_squared_error(y_val, y_pred_val)
test_rmse = np.sqrt(test_mse)
print(f"Val MSE scaled; {val_mse:.6f}")
print(f"Test MSE scaled: {test_mse:.6f}")
test_r2 = r2_score(y_test, y_pred)
val_r2 = r2_score(y_val, y_pred_val)
print(f"Val R2 scaled: {val_r2:.6f}")
print(f"Test R2 scaled: {test_r2:.6f}")
baseline_mse = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))
print(f"Baseline (train mean) MSE: {baseline_mse:.6f}")
print(f"Prediction spread: std={y_pred.std():.6f} vs y_test std={y_test.std()}")

y_test_numpy = y_test.to_numpy()
y_pred_numpy = y_pred

results_dict = {
    'best_estimator': f'NN_Bayesian{suffix}.keras',
    'scaled_metrics': {'mse': test_mse, 'r2': test_r2, "baseline_mse": baseline_mse},
    'y_test_scaled': y_test_numpy.flatten(),
    'y_pred_scaled': y_pred_numpy.flatten(),
}

pickle_file = f'NN_results{suffix}.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'results_dict dumped in {pickle_file}!')
