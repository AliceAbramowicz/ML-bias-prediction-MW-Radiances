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

X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_stats_scaled.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_stats_scaled.csv")
y_train = pd.read_csv("/perm/nld3863/create_dataset_all/y_train_stats_scaled.csv")
y_test = pd.read_csv("/perm/nld3863/create_dataset_all/y_test_stats_scaled.csv")

print("y train scaled", y_train)
print("X train scaled:", X_train)

def build_model(hp):
    model = keras.Sequential()
    
    batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256, 512, 1024])
    
    units1 = hp.Int('units1', min_value=10, max_value=360, step=25)
    units2 = hp.Int('units2', min_value=10, max_value=360, step=25)
    units3 = hp.Int('units3', min_value=10, max_value=360, step=25)
    units4 = hp.Int('units4', min_value=10, max_value=360, step=25)
    units5 = hp.Int('units5', min_value=10, max_value=360, step=25)
    model.add(keras.layers.Dense(units=units1, activation=hp.Choice("activation1", ["relu", "tanh"]), input_shape=X_train.shape[1:]))
    if hp.Boolean('dropout1'):
        model.add(layers.Dropout(rate=0.25))
    if hp.Boolean('add_layer_2'):
        model.add(keras.layers.Dense(units=units2, activation=hp.Choice("activation2", ["relu", "tanh"])))
    if hp.Boolean('dropout2'):
        model.add(layers.Dropout(rate=0.25))
    if hp.Boolean('add_layer_3'):
        model.add(keras.layers.Dense(units=units3, activation=hp.Choice("activation3", ["relu", "tanh"])))
    if hp.Boolean('dropout3'):
        model.add(layers.Dropout(rate=0.25))
    if hp.Boolean('add_layer_4'):
        model.add(keras.layers.Dense(units=units4, activation=hp.Choice("activation4", ["relu", "tanh"])))
    if hp.Boolean('dropout4'):
        model.add(layers.Dropout(rate=0.25))
    if hp.Boolean('add_layer_5'):
        model.add(keras.layers.Dense(units=units5, activation=hp.Choice("activation5", ["relu", "tanh"])))
    if hp.Boolean('dropout5'):
        model.add(layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(1))
    
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])
    hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05])
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

# Keras Tuner library for hyperparameter optimization
tuner = kt.BayesianOptimization(build_model,
                     objective='val_loss',
                     max_trials=10,
                     alpha=0.0001,
                     seed=100,
                     tune_new_entries=True) 

stop_early = EarlyStopping(
    monitor='val_loss',
    patience=8,
)

# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=200,  batch_size = 128, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units1')} with the activation function {best_hps.get('activation1')}. 
The optimal amount of hidder layers includes a 2nd layer: {best_hps.get('add_layer_2')}.
The optimal amount of hidder layers includes a 3rd layer: {best_hps.get('add_layer_3')}.
The optimal amount of hidder layers includes a 4th layer: {best_hps.get('add_layer_4')}.
The optimal amount of hidder layers includes a 5th layer: {best_hps.get('add_layer_5')}.
The optimal batch size is {best_hps.get('batch_size')}.
The best set-up includes dropouts: {best_hps.get('dropout1')}, {best_hps.get('dropout2')}, {best_hps.get('dropout3')}, {best_hps.get('dropout4')}, {best_hps.get('dropout5')}.
The best optimizer is {best_hps.get('optimizer')}.
The best learning rate is {best_hps.get('learning_rate')}.
""")

if best_hps.get('add_layer_2'):
    print(f"The optimal activation function for the 2nd hidden layer is {best_hps.get('activation2')}. The optimal number of units in the second densely-connected layer is {best_hps.get('units2')}.")
if best_hps.get('add_layer_3'):
    print(f"The optimal activation function for the 3rd hidden layer is {best_hps.get('activation3')}. The optimal number of units in the third densely-connected layer is {best_hps.get('units3')}.")
if best_hps.get('add_layer_4'):
    print(f"The optimal activation function for the 4th hidden layer is {best_hps.get('activation4')}. The optimal number of units in the 4th densely-connected layer is {best_hps.get('units4')}.")
if best_hps.get('add_layer_5'):
    print(f"The optimal activation function for the 5th hidden layer is {best_hps.get('activation5')}. The optimal number of units in the 5th densely-connected layer is {best_hps.get('units5')}.")

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[stop_early])
y_pred = model.predict(X_test)

mse_scaled = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) scaled: {mse_scaled}")
r2_scaled = r2_score(y_test, y_pred)
print(f"R-squared (R2) scaled: {r2_scaled}")

model.save('NN_final_model_BayesianOpt.keras')
with open('training_history_BayesianOpt.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Final training history saved.")

results_dict = {
    'final_model_path': 'NN_final_model_BayesianOpt.keras',
    'training_history_path': 'training_history_BayesianOpt.pkl',
    'scaled_metrics': {'mse': mse_scaled, 'r2': r2_scaled},
    'y_test_scaled': y_test,
    'y_pred_scaled': y_pred,
}

pickle_file = 'NN_results_BayesianOpt.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(results_dict, f)

print(f"Final model saved in {pickle_file}")
