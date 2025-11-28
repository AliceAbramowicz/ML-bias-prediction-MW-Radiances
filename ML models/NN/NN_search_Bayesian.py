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

tf.random.set_seed(100)
np.random.seed(100)
random.seed(100)

X_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_train_stats_scaled.csv")
X_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/X_test_stats_scaled.csv")
y_train = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_train_stats_scaled.csv")
y_test = pd.read_csv("/perm/nld3863/pipeline_datasets/datasets/final_df/y_test_stats_scaled.csv")

print("y train scaled", y_train.head(3))
print("X train scaled:", X_train.head(3))

print(X_train.dtypes)
print(y_train.dtypes)


# Limit CPU threads to avoid pthread_create errors
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

def build_model(hp):
    model = keras.Sequential()
    
    batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256, 512])
    l2_reg = hp.Float('l2_reg', 1e-5, 1e-4, sampling='log') # tune for lambda of the Ridge penalty (L2 regularization)
    use_l2 = hp.Boolean('use_l2_regularization')
    units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    kernel_regularizer = keras.regularizers.l2(l2_reg) if use_l2 else None
    kernel_init = hp.Choice('kernel_initializer', ['he_normal', 'he_uniform'])
    model.add(layers.Dense(units=units1, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_init, input_shape=X_train.shape[1:]))

    if hp.Boolean('batch_norm1'):
        model.add(layers.BatchNormalization()) # recenter and rescale the batch
    model.add(layers.Activation("relu"))
    
    if hp.Boolean('dropout1'):
        model.add(layers.Dropout(rate=0.25))

    for i in range(2, 4):  
        if hp.Boolean(f'add_layer_{i}'):
            model.add(layers.Dense(units=hp.Int(f'units{i}', min_value=32, max_value=320, step=32), kernel_regularizer=kernel_regularizer)) 
            if hp.Boolean(f'batch_norm{i}'):
                model.add(layers.BatchNormalization())
            model.add(layers.Activation("relu"))
            if hp.Boolean(f'dropout{i}'):
                model.add(layers.Dropout(rate=0.25))
    
    model.add(keras.layers.Dense(1))

    learning_rate = hp.Float('adam_learning_rate', min_value=1e-4, max_value=1e-1, sampling='log')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
 
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

# Keras Tuner library for hyperparameter optimization
tuner = kt.BayesianOptimization(build_model,
                     objective='val_loss',
                     max_trials=25,
                     alpha=0.0001,
                     beta=2.6,
                     seed=100,
                     tune_new_entries=True,
                     project_name='bayesian_search_Q_T')

stop_early = EarlyStopping(
    monitor='val_loss',
    patience=12,
)

split_index = int(len(X_train) * 0.85)

X_val = X_train[split_index:]
y_val = y_train[split_index:]

X_train = X_train[:split_index]
y_train = y_train[:split_index]

# IF YOU NEED TO SEARCH FOR HYPERPARAM:
# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=125, validation_data=(X_val, y_val), callbacks=[stop_early], shuffle=False)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_batch_size = best_hps.get('batch_size')


stop_early_ = EarlyStopping(
    monitor='val_loss',
    patience=75, #more patience for training than hyperparam search
    restore_best_weights=True)

# IF YOU NEED TO TRAIN YOUR MODEL:
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=1000, batch_size=best_batch_size, validation_data=(X_val, y_val), shuffle=False, callbacks=[stop_early_])
with open('HISTORY_Bayesian_Q_T.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save("NN_Bayesian_Q_T.keras")


# IF MODEL ALREADY TRAINED, LOAD IT:
# model = tf.keras.models.load_model('FINAL_NN.keras')

print("model's architecture:")
model.summary()

y_pred = model.predict(X_test)

mse_scaled = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) scaled: {mse_scaled}")
r2_scaled = r2_score(y_test, y_pred)
print(f"R-squared (R2) scaled: {r2_scaled}")

y_test_numpy = y_test.to_numpy()
y_pred_numpy = y_pred

results_df = pd.DataFrame({
    'index': X_test.index,
    'y_test_scaled': y_test_numpy.flatten(),
    'y_pred_scaled': y_pred_numpy.flatten()
})

results_dict = {
    'final_model_path': 'NN.keras',
    'scaled_metrics': {'mse': mse_scaled, 'r2': r2_scaled},
    'results_df': results_df,
    'y_test_scaled': y_test,
    'y_pred_scaled': y_pred
}

pickle_file = 'results_NN_Q_T'
with open(pickle_file, 'wb') as f:
    pickle.dump(results_dict, f)

print('results_dict dumped!')
