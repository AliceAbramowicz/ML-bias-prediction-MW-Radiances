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

X_train = pd.read_csv("/perm/nld3863/create_dataset_all/X_train_exp_stats_scaled.csv")
X_test = pd.read_csv("/perm/nld3863/create_dataset_all/X_test_exp_stats_scaled.csv")
y_train = pd.read_csv("/perm/nld3863/create_dataset_all/y_train_stats_scaled.csv")
y_test = pd.read_csv("/perm/nld3863/create_dataset_all/y_test_stats_scaled.csv")

print("y train scaled", y_train.head(3))
print("X train scaled:", X_train.head(3))

def build_model(hp):
    model = keras.Sequential()
    
    batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256, 512])
    
    units1 = hp.Int('units1', min_value=32, max_value=512, step=32)

    model.add(keras.layers.Dense(units=units1, activation=hp.Choice("activation1", ["relu", "tanh"]), kernel_regularizer=keras.regularizers.l2(hp.Float('l2_reg1', 1e-5, 1e-2, sampling='log')), input_shape=X_train.shape[1:]))
    if hp.Boolean('batch_norm1'):
        model.add(layers.BatchNormalization()) # recenter and rescale the batch
    if hp.Boolean('dropout1'):
        model.add(layers.Dropout(rate=hp.Float('dropout_rate1', 0.1, 0.5, step=0.05)))

    for i in range(2, 6):  
        if hp.Boolean(f'add_layer_{i}'):
            model.add(layers.Dense(
                units=hp.Int(f'units{i}', min_value=32, max_value=512, step=32),
                activation=hp.Choice(f'activation{i}', ["relu", "tanh"]),
                kernel_regularizer=keras.regularizers.l2(hp.Float(f'l2_reg{i}', 1e-5, 1e-2, sampling='log'))
            ))
            if hp.Boolean(f'batch_norm{i}'):
                model.add(layers.BatchNormalization())
            if hp.Boolean(f'dropout{i}'):
                model.add(layers.Dropout(rate=hp.Float(f'dropout_rate{i}', 0.1, 0.5, step=0.05)))
    
    model.add(keras.layers.Dense(1))
    
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'RMSprop', 'adamw'])
    if optimizer_choice == 'adam':
        learning_rate = hp.Float('adam_learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    elif optimizer_choice == 'sgd':
        learning_rate = hp.Float('sgd_learning_rate', min_value=1e-3, max_value=1e-1, sampling='log')
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    elif optimizer_choice == 'RMSprop':
        learning_rate = hp.Float('rmsprop_learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    elif optimizer_choice == 'adamw':
        learning_rate = hp.Float('adamw_learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_choice}")
 
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

# Keras Tuner library for hyperparameter optimization
tuner = kt.BayesianOptimization(build_model,
                     objective='val_loss',
                     max_trials=20,
                     alpha=0.0001,
                     beta=3,
                     seed=100,
                     tune_new_entries=True) 

stop_early = EarlyStopping(
    monitor='val_loss',
    patience=12,
)

# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_batch_size = best_hps.get('batch_size')

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units1')} with the activation function {best_hps.get('activation1')}. 
The optimal amount of hidder layers includes a 2nd layer: {best_hps.get('add_layer_2')}.
The optimal amount of hidder layers includes a 3rd layer: {best_hps.get('add_layer_3')}.
The optimal amount of hidder layers includes a 4th layer: {best_hps.get('add_layer_4')}.
The optimal amount of hidder layers includes a 5th layer: {best_hps.get('add_layer_5')}.
The optimal batch size is {best_hps.get('batch_size')}.
The best set-up includes dropouts: {best_hps.get('dropout1')}, {best_hps.get('dropout2')}, {best_hps.get('dropout3')}, {best_hps.get('dropout4')}, {best_hps.get('dropout5')}.
The best optimizer is {best_hps.get('optimizer')}.
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
history = model.fit(X_train, y_train, epochs=250, batch_size=best_batch_size, validation_split=0.2, callbacks=[stop_early])
y_pred = model.predict(X_test)

mse_scaled = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) scaled: {mse_scaled}")
r2_scaled = r2_score(y_test, y_pred)
print(f"R-squared (R2) scaled: {r2_scaled}")

model.save('NN_final_model_BayesianOpt_exp_all.keras')
with open('training_history_BayesianOpt_exp_all.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("Final training history saved.")

results_dict = {
    'final_model_path': 'NN_final_model_BayesianOpt_exp_all.keras',
    'training_history_path': 'training_history_BayesianOpt_exp_all.pkl',
    'scaled_metrics': {'mse': mse_scaled, 'r2': r2_scaled},
    'y_test_scaled': y_test,
    'y_pred_scaled': y_pred,
}

pickle_file = 'NN_results_BayesianOpt_exp_all.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(results_dict, f)

print(f"Final model saved in {pickle_file}")
