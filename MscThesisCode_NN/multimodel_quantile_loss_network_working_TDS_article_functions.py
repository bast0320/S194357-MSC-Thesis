
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
from FFNN_as_basis_to_TAQR import * # Standard FFNN Class, and a function "train_ffnn" to train the FFNN.
from functions_for_TAQR import * # Jan's algo in Python, currently in a very raw state 4/4-24. 
from helper_functions import * # Helper functions: "generate_spline_matrices"
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
import pandas as pd
from FFNN_as_basis_to_TAQR import * # Standard FFNN Class, and a function "train_ffnn" to train the FFNN.

from functions_for_TAQR import * # Jan's algo in Python, currently in a very raw state 4/4-24. 

from helper_functions import * # Helper functions: "generate_spline_matrices"


def calculate_error(q, y_true, y_pred):
    # y_true = tf.linspace(10.0, 20.0, 100)
    # y_pred = tf.linspace(10.0, 20.0, 200)
    loss = tf.map_fn(lambda y_pred: quantile_loss_2(q, y_true, y_pred), y_pred, fn_output_signature=tf.float32)
    loss = tf.reduce_mean(loss, axis=1)
    return y_true, y_pred, loss
 

import tensorflow_probability as tfp
import tensorflow as tf


    
class QuantileRegression(tf.keras.Model):
    def __init__(self, n_quantiles: int):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(50, activation='relu')
        self.quantile_outputs = []
        for i in range(n_quantiles):
            self.quantile_outputs.append(tf.keras.layers.Dense(1)) # here depends on how much we want out per quantile...

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        quantile_outputs = []
        for output_layer in self.quantile_outputs:
            quantile_outputs.append(output_layer(x))
        return tf.concat(quantile_outputs, axis=1)


class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        loss = tf.map_fn(lambda q: quantile_loss_2(q, y_true, y_pred), self.quantiles, fn_output_signature=tf.float32)
        return tf.reduce_sum(loss, axis=-1)  # sum or mean here. I think sum is correct

# def train_model(quantiles, epochs: int, lr: float, batch_size: int, x, y):
#     model = QuantileRegression(n_quantiles=len(quantiles))
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
#     @tf.function
#     def train_step(x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             output = model(x_batch, training=True)
#             losses = []
#             for i, q in enumerate(quantiles):
#                 loss = quantile_loss_2(q, y_true = y_batch, y_pred = output[:, i]) # tfp.stats.percentile(y_batch,100*q)
#                 losses.append(loss)
#             total_loss = tf.reduce_mean(losses)
        
#         gradients = tape.gradient(total_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
 
#         return total_loss
    
#     for epoch in range(epochs):
#         dataset = tf.data.Dataset.from_tensor_slices((x, y))
#         dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size, drop_remainder=True)
#         # dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size).repeat()
#         # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#         # dataset = tf.data.Dataset.padded_batch(dataset, batch_size=50)
#         epoch_loss = 0.0
       
#         j = 0
#         for x_batch, y_batch in dataset:
#             #print("shapes: ", x_batch.shape, y_batch.shape, "epoch: ", epoch, "step: ", j)
#             batch_loss = train_step(x_batch, y_batch)
#             epoch_loss += batch_loss
#             j +=1
        
        
#         epoch_loss /= len(dataset)
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}, Loss: {epoch_loss.numpy()}")
    
#     return model

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def map_range(values, a, b, c, d):
    # Map values from range [a, b] to range [c, d]
    return (values - a) / (b - a) * (d - c) + c

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

def train_model(quantiles, epochs: int, lr: float, batch_size: int, x, y, x_val, y_val, data_name):
    model = QuantileRegression(n_quantiles=len(quantiles))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            output = model(x_batch, training=True)
            losses = []
            for i, q in enumerate(quantiles):
                loss = quantile_loss_2(q, y_true=y_batch, y_pred=output[:, i])
                losses.append(loss)
            total_loss = tf.reduce_mean(losses)
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss
    
    @tf.function
    def val_step(x_batch, y_batch):
        output = model(x_batch, training=False)
        losses = []
        for i, q in enumerate(quantiles):
            loss = quantile_loss_2(q, y_true=y_batch, y_pred=output[:, i])
            losses.append(loss)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    train_loss_history = []
    val_loss_history = []
    y_preds = []
    y_true = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        num_batches = 0
        
        # Training loop
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size, drop_remainder=True)
        
        for x_batch, y_batch in dataset:
            # print("shapes, training: ", x_batch.shape, y_batch.shape)
            batch_train_loss = train_step(x_batch, y_batch)
            epoch_train_loss += batch_train_loss
            num_batches += 1
            
            y_preds.append(model(x_batch, training=False))
            y_true.append(y_batch)
        
        epoch_train_loss /= num_batches
        train_loss_history.append(epoch_train_loss)
        
        # Validation loop
        num_val_batches = 0
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size, drop_remainder=True)
        # val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
        for x_val_batch, y_val_batch in val_dataset:
            
            print("shapes, validation: ", x_val_batch.shape, y_val_batch.shape)
            batch_val_loss = val_step(x_val_batch, y_val_batch)
            epoch_val_loss += batch_val_loss
            num_val_batches += 1
        
        epoch_val_loss /= num_val_batches
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}")

    y_preds_concat = tf.concat(y_preds, axis=0).numpy()
    y_true_concat = tf.concat(y_true, axis=0).numpy()
    
    print("shape y_preds_concat: ", y_preds_concat.shape)
    print("shape y_true_concat: ", y_true_concat.shape)
    rmse = np.sqrt(np.mean((y_true_concat[:, 25] - y_preds_concat[:, 10])**2))

    # Plotting in a 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training Analysis', fontsize=16)

    # Training and Validation Loss Curve
    axs[0].plot(range(1, epochs+1), train_loss_history, label='Training Loss', color="black")
    axs[0].plot(range(1, epochs+1), val_loss_history, label='Validation Loss', color="blue")
    axs[0].scatter(range(1, epochs+1), train_loss_history, color="black")
    axs[0].scatter(range(1, epochs+1), val_loss_history, color="blue")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Curve')
    axs[0].grid()
    axs[0].legend()

    # Predicted vs Actuals with RMSE
    vals_each_side = 200
    n = len(y_true_concat)
    vals_per_epoch = int(n / epochs)

    rows = np.concatenate((np.arange(int(vals_per_epoch/2), int(vals_per_epoch/2)+vals_each_side),  np.arange(n-vals_each_side, n)))
    cols_true = np.array([15,25,35])
    cols_preds = map_range(cols_true, 0, 53, 0, 20)

    y_true_plot = y_true_concat[np.ix_(rows, cols_true)]
    y_preds_plot = y_preds_concat[np.ix_(rows, cols_preds)]

    axs[1].plot(y_true_plot, label='True', color='black', alpha=0.6)
    axs[1].plot(y_preds_plot, label='Predictions', color='blue', alpha=0.6)
    axs[1].axvline(x=vals_each_side, color='r', linestyle='--', linewidth=2)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].set_title(f'Predicted vs Actuals')
    axs[1].text(0.1, 0.05, f'Epoch 1', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
    axs[1].text(0.9, 0.05, f'Last Epoch', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
    legend_without_duplicate_labels(axs[1])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"train_nn_plots/Training_NN_{data_name}.pdf")
    plt.show()

    return model


def variogram_score_R(x, y, p=0.5):
    """
    Calculate the Variogram score for a given quantile.
    From the paper Energy and AI, Mathias
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0

    print("m,k: ", m, k)
    # Iterate through all pairs
    for i in range(k - 1):
        for j in range(i + 1, k):
            Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j])**p)
            score += (np.abs(y[i] - y[j])**p - Ediff)**2

    # Variogram score
    return score




def single_quantile_skill_score(y_true, y_pred, quantile):
    """
    Calculate the Quantile Skill Score (QSS) for quantile forecasts.

    Parameters:
    y_true (numpy.array or list): True observed values. 1D array.
    y_pred (numpy.array or list): Predicted quantile values. 1D array.
    quantile (float): Quantile level, between 0 and 1.

    Returns:
    float: The QSS for the given quantile forecasts.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must be the same length"
    assert 0 <= quantile <= 1, "quantile must be between 0 and 1"

    N = len(y_true)
    q = quantile
    score = 0

    for y, y_hat in zip(y_true, y_pred):
        if y >= y_hat:
            score += q * np.abs(y - y_hat)
        else:
            score += (1 - q) * np.abs(y - y_hat)

    return score / N 

def multi_quantile_skill_score(y_true, y_pred, quantiles):
    """
    Calculate the Quantile Skill Score (QSS) for multiple quantile forecasts.

    y_true: This is a 1D numpy array or list of the true observed values.
    y_pred: This is a 2D numpy array or list of lists of the predicted quantile values. The outer dimension should be the same length as y_true, and the inner dimension should be the number of quantiles.
    quantiles: This is a 1D numpy array or list of the quantile levels. It should be the same length as the inner dimension of y_pred.

    Parameters:
    y_true (numpy.array or list): True observed values. 1D array.
    y_pred (numpy.array or list of lists): Predicted quantile values. 2D array.
    quantiles (numpy.array or list): Quantile levels, between 0 and 1. 1D array.

    Returns:
    numpy.array: The QSS for each quantile forecast. 1D array.
    """
    # Convert y_pred to a numpy array
    y_pred = np.array(y_pred)

    if y_pred.shape[0] > y_pred.shape[1]:
        y_pred = y_pred.T

    # print("shape y_pred: ", y_pred.shape)   
    # print("shape y_true: ", y_true.shape)
    # assert len(y_true) == len(y_pred[0]), "y_true and y_pred must be the same length" TODO: Should be kept active,... 
    assert all(0 <= q <= 1 for q in quantiles), "All quantiles must be between 0 and 1"
    assert len(quantiles) == len(y_pred), "Number of quantiles must match inner dimension of y_pred"

    N = len(y_true)
    scores = np.zeros(len(quantiles))

    for y, y_hats in zip(y_true, y_pred):
        for i, (y_hat, q) in enumerate(zip(y_hats, quantiles)):
            if y >= y_hat:
                scores[i] += q * np.abs(y - y_hat)
            else:
                scores[i] += (1 - q) * np.abs(y - y_hat)

    return scores / N



#### LSTM VERSION of the network.... 30/4-24
import tensorflow as tf

class QuantileRegressionLSTM(tf.keras.Model):
    def __init__(self, n_quantiles, units, n_timesteps):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(units, input_shape=(None, n_quantiles, n_timesteps), return_sequences=False)
        # self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(n_quantiles, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(n_quantiles, activation='relu')
        self.n_quantiles = n_quantiles
        self.n_timesteps = n_timesteps

    def call(self, inputs, training=None):
        x = self.lstm(inputs, training=training)
        # x = self.layer_norm(x)
        x = self.dense(x)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super(QuantileRegressionLSTM, self).get_config()
        config.update({
            'n_quantiles': self.n_quantiles,
            'units': self.lstm.units,
            'n_timesteps': self.n_timesteps,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# class QuantileRegressionLSTM(tf.keras.Model):
#     def __init__(self, n_quantiles, units, n_timesteps, **kwargs):
#         # Extract custom arguments here
#         trainable = kwargs.pop('trainable', True)
#         super(QuantileRegressionLSTM, self).__init__(**kwargs)
#         self.trainable = trainable
#         self.lstm = tf.keras.layers.LSTM(units, input_shape=(None, n_timesteps), return_sequences=False)
#         # self.layer_norm = tf.keras.layers.LayerNormalization()
#         self.dense = tf.keras.layers.Dense(n_quantiles, activation='sigmoid')
#         self.dense2 = tf.keras.layers.Dense(n_quantiles, activation='relu')
#         self.n_quantiles = n_quantiles
#         self.n_timesteps = n_timesteps

#     def call(self, inputs, training=None):
#         x = self.lstm(inputs, training=self.trainable)
#         # x = self.layer_norm(x)
#         x = self.dense(x)
#         x = self.dense2(x)
#         return x

    # def get_config(self):
    #     config = super(QuantileRegressionLSTM, self).get_config()
    #     config.update({
    #         'n_quantiles': self.n_quantiles,
    #         'units': self.lstm.units,
    #         'n_timesteps': self.n_timesteps,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

def quantile_loss_3(q, y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tfp.stats.percentile(y_true, 100*q, axis = 1)
    # y_pred = tfp.stats.percentile(y_pred, 100*q, axis = 1)

    error = y_true - y_pred
    return tf.maximum(q * error, (q - 1) * error) # tf.reduce_mean(, axis = -1)

def quantile_loss_func(quantiles):
    def loss(y_true, y_pred):
        losses = []
        for i, q in enumerate(quantiles):
            # print("activated? -------------****")
            # print("in quantile loss func the shapes are: ", y_true.shape, y_pred.shape)
            loss = quantile_loss_3(q, y_true, y_pred[:,  i]) #y_true[:,  i]
            losses.append(loss)
        return losses
    return loss


# This is a very good working alternative to the below.... 12/6-24. 
# def train_model_lstm(quantiles, epochs: int, lr: float, batch_size: int, x, y, n_timesteps):
#     model = QuantileRegressionLSTM(n_quantiles=len(quantiles), units=256, n_timesteps=n_timesteps)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
#     @tf.function
#     def train_step(x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             y_pred = model(x_batch, training=True)
#             losses = quantile_loss_func(quantiles)(y_batch, y_pred)
#             total_loss = tf.reduce_mean(losses)
#         grads = tape.gradient(total_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return total_loss
    
#     for epoch in range(epochs):
#         #print(f"Epoch {epoch+1}/{epochs}")
#         epoch_loss = 0.0
#         num_batches = 0
        
#         for i in range(0, len(x), batch_size):
#             x_batch = x[i:i+batch_size]
#             y_batch = y[i:i+batch_size]
            
#             batch_loss = train_step(x_batch, y_batch)
#             epoch_loss += batch_loss
#             num_batches += 1
        
#         epoch_loss /= num_batches

#         # if epoch % 50 == 0:
#         print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
#     return model
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def map_range(values, input_start, input_end, output_start, output_end):

    mapped_values = []
    for value in values:
        # Calculate the proportion of value in the input range
        proportion = (value - input_start) / (input_end - input_start)
        
        # Map the proportion to the output range
        mapped_value = output_start + (proportion * (output_end - output_start))
        mapped_values.append(int(mapped_value))
    
    return np.array(mapped_values)

# def train_model_lstm(quantiles, epochs: int, lr: float, batch_size: int, x, y, n_timesteps):
#     model = QuantileRegressionLSTM(n_quantiles=len(quantiles), units=256, n_timesteps=n_timesteps)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    

#     # @tf.function # function working great, but we don't need to output the grads and weights, so we can outcomment this.
#     # def train_step(x_batch, y_batch):
#     #     with tf.GradientTape() as tape:
#     #         y_pred = model(x_batch, training=True)
#     #         losses = quantile_loss_func(quantiles)(y_batch, y_pred)
#     #         total_loss = tf.reduce_mean(losses)
#     #     grads = tape.gradient(total_loss, model.trainable_variables)
#     #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     #     print("grads shape: ", grads)
#     #     print("model trainable variables shape: ", model.trainable_variables)
#     #     return total_loss, grads, model.trainable_variables
    

#     @tf.function
#     def train_step(x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             y_pred = model(x_batch, training=True)
#             losses = quantile_loss_func(quantiles)(y_batch, y_pred)
#             total_loss = tf.reduce_mean(losses)
        
#         grads = tape.gradient(total_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
#         # Calculate the norm of the gradients
#         grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
        
#         # Calculate the sum of the trainable variables
#         trainable_vars_sum = tf.reduce_sum([tf.reduce_sum(v) for v in model.trainable_variables])
        
#         # print("Gradient norm: ", grad_norm)
#         # print("Sum of trainable variables: ", trainable_vars_sum)
        
#         return total_loss, grad_norm, trainable_vars_sum
    
#     loss_history = []
#     grads_history = []
#     weights_history = []
#     y_preds = []
#     y_true = []

#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         num_batches = 0
#         grad_every_loop = 0
#         weights_every_loop = 0
        
#         for i in range(0, len(x), batch_size):
#             x_batch = x[i:i+batch_size]
#             y_batch = y[i:i+batch_size]
            
#             batch_loss, grads, weights = train_step(x_batch, y_batch)
#             # print("model summary: ", model.summary())
#             epoch_loss += batch_loss
#             num_batches += 1

#             # print("grads shape: ", grads, "and grads reduced: ", tf.reduce_sum(grads).shape)
#             # print("weights shape: ", weights, "and weights reduced: ", tf.reduce_sum(weights).shape)
#             grad_every_loop += grads.numpy()
#             weights_every_loop += weights.numpy()
            
#             y_preds.append(model(x_batch, training=False))
#             y_true.append(y_batch)

#         grads_history.append((grad_every_loop) )
#         weights_history.append((weights_every_loop) )
        
#         epoch_loss /= num_batches
#         loss_history.append(epoch_loss)

#         print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

#         # List to store layer details
#         # layer_details = []

#         # # Loop through each layer to get weights and add to the list
#         # for i, layer in enumerate(model.layers):
#         #     layer_name = layer.name
#         #     weights = layer.get_weights()
            
#         #     for j, weight in enumerate(weights):
#         #         weight_type = f"Weight {j + 1}"
#         #         weight_shape = weight.shape
#         #         layer_details.append([layer_name, weight_type, weight_shape])

#         # # Convert to DataFrame
#         # df = pd.DataFrame(layer_details, columns=["Layer Name", "Weight Type", "Shape"])

#         # # Convert DataFrame to LaTeX code
#         # latex_code = df.to_latex(index=False)

#         # # Save the LaTeX code to a .tex file
#         # output_path = "layer_weights.tex"
#         # with open(output_path, "w") as f:
#         #     f.write(latex_code)

#         # print(f"LaTeX code has been saved to {output_path}")


#     # y_preds_concat =  np.array(y_preds)
#     # y_true_concat =  np.array(y_true)

#     y_preds_concat =  tf.concat(y_preds, axis=0).numpy()
#     y_true_concat =  tf.concat(y_true, axis=0).numpy()
#     # print the shapes
#     print("shape y_preds_concat: ", y_preds_concat.shape)
#     print("shape y_true_concat: ", y_true_concat.shape)
#     rmse = np.sqrt(np.mean((y_true_concat[:,25] - y_preds_concat[:,10])**2))

#     # Plotting in a 2x2 grid
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle('Model Training Analysis', fontsize=16)

#     # Training Loss Curve
#     axs[0, 0].plot(range(1, epochs+1), loss_history, label='Training Loss', color = "black")
#     axs[0, 0].scatter(range(1, epochs+1), loss_history, color = "black")
#     axs[0, 0].set_xlabel('Epochs')
#     axs[0, 0].set_ylabel('Loss')
#     axs[0, 0].set_title('Training Loss Curve')
#     # add grid
#     axs[0, 0].grid()
#     # axs[0, 0].legend()

#     # Gradient Histogram
#     axs[0, 1].plot(range(1, epochs+1),grads_history,   color='black')
#     axs[0, 1].scatter(range(1, epochs+1),grads_history,   color='black')
#     axs[0, 1].set_xlabel('Epochs')
#     axs[0, 1].set_ylabel('Gradient values')
#     axs[0, 1].set_title('Gradient Values over Epochs')
#     axs[0, 1].grid()

#     # Weight Histogram
#     axs[1, 0].plot(range(1, epochs+1),weights_history,  color='black')
#     axs[1, 0].scatter(range(1, epochs+1),weights_history,  color='black')
#     axs[1, 0].set_xlabel('Epochs')
#     axs[1, 0].set_ylabel('Weight values')
#     axs[1, 0].set_title('Weight Values over Epochs')
#     axs[1, 0].grid()

#     # Predicted vs Actuals with RMSE
#     vals_each_side = 200
#     n = len(y_true_concat)
#     vals_per_epoch = int(n / epochs)

#     rows = np.concatenate((np.arange(int(vals_per_epoch/2), int(vals_per_epoch/2)+vals_each_side),  np.arange(n-vals_each_side, n) ))
#     cols_true = np.array([15,25,35])
#     cols_preds = map_range(cols_true, 0, 53, 0, 20)

#     # Use np.ix_ to generate the cross-product of row and column indices
#     y_true_plot = y_true_concat[np.ix_(rows, cols_true)]
#     y_preds_plot = y_preds_concat[np.ix_(rows, cols_preds)]

#     axs[1, 1].plot(y_true_plot, label='True', color='black', alpha=0.6)
#     axs[1, 1].plot(y_preds_plot, label='Predictions', color='blue', alpha=0.6)
#     # add a h line vals_each_side
#     axs[1, 1].axvline(x=vals_each_side, color='r', linestyle='--', linewidth=2)
#     axs[1, 1].set_xlabel('Time')
#     axs[1, 1].set_ylabel('Values')
#     axs[1, 1].set_title(f'Predicted vs Actuals') # (RMSE: {rmse:.4f})')
#     # add text upper left "Epoch 1"
#     axs[1, 1].text(0.1, 0.05, f'Epoch 1', horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
#     # add text upper right "Last Epoch"
#     axs[1, 1].text(0.9, 0.05, f'Last Epoch', horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
#     legend_without_duplicate_labels(axs[1, 1])

#     # Adjust layout to prevent overlap
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()
    
    
#     return model

def train_model_lstm(quantiles, epochs: int, lr: float, batch_size: int, x, y, x_val, y_val, n_timesteps, data_name):
    model = QuantileRegressionLSTM(n_quantiles=len(quantiles), units=256, n_timesteps=n_timesteps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x_batch, y_batch):
        # IF WE WANT I THINK WE HERE COULD RUN THE TAQR ALGORITHM AND THEN USE THAT LOSS TO OPTIMIZE THE NETWORK FOR AMAZING BASIS FUNCTIONS...
        # THEN OF COURSE WE WOULD NEED TO WRITE THEORY ON HOW THE GRADIENT TAPE WORKS, BUT IF WE CHANGE Y_BATCH AND X_BATCH OUTSIDE OF THIS TRAINING LOOP,
        # THEN WE POTENTIALLY COULD USE THE TAQR ALGO ON THE X BATCH AND THROUGH EPOCHS WE WOULD CHANGE THE MODEL, BUT NOT THE TAQR ALGO
        # SO WE SEE HOW CHANGING THE MODEL AFFECTS THE 
        # IT BREAKS DOWN... DOESN'T IT... SINCE THE MODEL IT OUTPUTTING THE NEW basis, and we don't really know what we should COMPARE it too.... 
        # unless we of course compare it to splines, but then we should just do that...
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            losses = quantile_loss_func(quantiles)(y_batch, y_pred)
            total_loss = tf.reduce_mean(losses)
        
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss
    
    @tf.function
    def val_step(x_batch, y_batch):
        y_pred = model(x_batch, training=False)
        losses = quantile_loss_func(quantiles)(y_batch, y_pred)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    train_loss_history = []
    val_loss_history = []
    y_preds = []
    y_true = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        num_batches = 0
        
        # Training loop
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            batch_train_loss = train_step(x_batch, y_batch)
            epoch_train_loss += batch_train_loss
            num_batches += 1
            
            y_preds.append(model(x_batch, training=False))
            y_true.append(y_batch)
        
        epoch_train_loss /= num_batches
        train_loss_history.append(epoch_train_loss)
        
        # Validation loop
        num_val_batches = 0
        for i in range(0, len(x_val), batch_size):
            x_val_batch = x_val[i:i+batch_size]
            y_val_batch = y_val[i:i+batch_size]
            
            batch_val_loss = val_step(x_val_batch, y_val_batch)
            epoch_val_loss += batch_val_loss
            num_val_batches += 1
        
        epoch_val_loss /= num_val_batches
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}")

    y_preds_concat = tf.concat(y_preds, axis=0).numpy()
    y_true_concat = tf.concat(y_true, axis=0).numpy()
    
    print("shape y_preds_concat: ", y_preds_concat.shape)
    print("shape y_true_concat: ", y_true_concat.shape)
    rmse = np.sqrt(np.mean((y_true_concat[:,25] - y_preds_concat[:,10])**2))

    # Plotting in a 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training Analysis', fontsize=16)

    # Training and Validation Loss Curve
    axs[0].plot(range(1, epochs+1), train_loss_history, label='Training Loss', color="black")
    axs[0].plot(range(1, epochs+1), val_loss_history, label='Validation Loss', color="blue")
    axs[0].scatter(range(1, epochs+1), train_loss_history, color="black")
    axs[0].scatter(range(1, epochs+1), val_loss_history, color="blue")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Curve')
    axs[0].grid()
    axs[0].legend()

    # Predicted vs Actuals with RMSE
    vals_each_side = 200
    n = len(y_true_concat)
    vals_per_epoch = int(n / epochs)

    rows = np.concatenate((np.arange(int(vals_per_epoch/2), int(vals_per_epoch/2)+vals_each_side),  np.arange(n-vals_each_side, n) ))
    cols_true = np.array([15,25,35])
    cols_preds = map_range(cols_true, 0, 53, 0, 20)

    y_true_plot = y_true_concat[np.ix_(rows, cols_true)]
    y_preds_plot = y_preds_concat[np.ix_(rows, cols_preds)]

    axs[1].plot(y_true_plot, label='True', color='black', alpha=0.6)
    axs[1].plot(y_preds_plot, label='Predictions', color='blue', alpha=0.6)
    axs[1].axvline(x=vals_each_side, color='r', linestyle='--', linewidth=2)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].set_title(f'Predicted vs Actuals')
    axs[1].text(0.1, 0.05, f'Epoch 1', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
    axs[1].text(0.9, 0.05, f'Last Epoch', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, bbox=dict(facecolor='white', edgecolor='white'))
    legend_without_duplicate_labels(axs[1])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"train_nn_plots/Training_NN_{data_name}.pdf")
    # plt.show()

    return model


###### LSTM v. 2 with gpt2-

import numpy as np

def create_dataset(X, Y, time_steps=10):

    if time_steps != 0:
        Xs, Ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i:(i + time_steps)]
            Xs.append(v)
            Ys.append(Y[i + time_steps -1]) # TODO check: trying minus 1 here to align... 
        return np.array(Xs), np.array(Ys)
    elif time_steps == 0:
        return X, Y
    
def create_dataset2(X, Y, time_steps):

    if time_steps != 0:
        Xs, Ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i:(i + time_steps)]
            Xs.append(v)
            Ys.append(Y[i + time_steps -1]) # TODO check: trying minus 1 here to align... 
        return np.array(Xs), np.array(Ys)
    elif time_steps == 0:
        return X, Y

# Assuming `data_x` and `data_y` are your original datasets:
# Reshape the input (X) and output (Y) data to fit the model
# timesteps_back_in_time = 10
# x, y = create_dataset(data_x, data_y, time_steps=timesteps_back_in_time)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# %%
import numpy as np

import numpy as np

def create_dataset3(X, Y, time_steps):
    '''
    This function takes in multidimensional X and array Y with equal length.
    Let us assume X takes the shape (50,1), then the output Xs would have shape (50, len(time_steps), 1)
    '''
    Xs, Ys = [], []
    for i in range(len(X)):
        X_entry = []
        for ts in time_steps:
            if i - ts >= 0:
                X_entry.append(X[i - ts])
            else:
                X_entry.append(np.zeros_like(X[0]))  # Padding with zeros for initial entries
        Xs.append(np.array(X_entry))
        Ys.append(Y[i])  # Current day's value
    return np.array(Xs), np.array(Ys)

# Example usage:
# X = np.arange(7).reshape(-1, 1)  # Sample data reshaped to (7, 1) for illustration
# Y = np.arange(7)  # Sample labels
# time_steps = [0, 1, 3]

# Xs, Ys = create_dataset3(X, Y, time_steps)
# print(Xs.shape)  # Should be (7, len(time_steps), 1)
# print(Ys.shape)  # Should be (7,)
# print(Xs)
# print(Ys)

# WORKS. 22/5-24. 


#%%



def build_model(n_quantiles, timesteps, features, lstm_units=50):
    model = Sequential([
        LSTM(lstm_units, input_shape=(timesteps, features), kernel_initializer='random_normal'),
        Dense(64, activation='relu'),
        Dense(n_quantiles)
    ])
    return model



def quantile_loss_2(q, y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tfp.stats.percentile(y_true, 100*q,axis = 1)

    # print("shape y_pred: ", y_pred.shape)
    # print("shape y_true: ", y_true.shape)

    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)

class QuantileLoss_2(tf.keras.losses.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        losses = []
        for i, q in enumerate(self.quantiles):
            loss = quantile_loss_2(q, y_true[:, i], y_pred[:, i])
            losses.append(loss)
        return tf.reduce_mean(losses)


######### BACKUP

# def quantile_loss_2(q, y_true, y_pred):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#     # y_true = tf.expand_dims(y_true, -1)  # This changes shape from [batch_size] to [batch_size, 1]

    

#     # plt.figure()
#     # plt.plot(y_true, label="y_true post")
#     # plt.legend()
#     # plt.show()

#     error = y_true - y_pred
#     return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)


# class QuantileLoss_2(tf.keras.losses.Loss):
#     def __init__(self, quantiles):
#         super().__init__()
#         self.quantiles = tf.constant(quantiles, dtype=tf.float32)

#     def call(self, y_true, y_pred):
#         # y_true is coming from the batch in each dataset, which currently is 50 long
#         print("shape y_pred: ", y_pred.shape)
#         print("shape y_true: ", y_true.shape)

     
#         loss = tf.map_fn(lambda q: quantile_loss_2(q, y_true, y_pred), self.quantiles, fn_output_signature=tf.float32)
#         # loss = quantile_loss_2(self.quantiles, y_true, y_pred)
#         return tf.reduce_sum(loss, axis=-1)
    
# def train_model_lstm_2(quantiles, epochs, lr, batch_size, x, y):
#     timesteps, features = x.shape[1], x.shape[2]
#     # features = x.shape[1]
    
#     # timesteps is coming from the create dataset function
#     # features, is the amount of ensembles, currently 52.

#     model = build_model(len(quantiles), timesteps, features, lstm_units= features)
#     print("model summary: ", model.summary())
#     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
#     loss_fn = QuantileLoss_2(quantiles)
    
#     @tf.function # IF WE OUTCOMMENT THIS, WE CAN DEBUG THE CODE IN FOR EXAMPLE quantile_loss_2-function.
#     def train_step2(x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             y_pred = model(x_batch, training=True)
#             loss = loss_fn(y_batch, y_pred)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         return loss
    

#     dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x)).batch(batch_size)
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for x_batch, y_batch in dataset:
#             batch_loss = train_step2(x_batch, y_batch)
#             # print(batch_loss.numpy()/len(dataset))
#             if np.isnan(batch_loss.numpy()):
#                 print("nan in batch loss")
#                 print("is there nans in x_batch? ", np.isnan(x_batch).any())
#                 print("is there nans in y_batch? ", np.isnan(y_batch).any())
#             else:
#                 epoch_loss += batch_loss.numpy()/len(dataset)

#         epoch_loss /= len(dataset)
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {epoch_loss}")

#     return model

# def train_model_lstm_2(quantiles, epochs, lr, batch_size, x, y):
#     timesteps, features = x.shape[1], x.shape[2]
#     model = build_model(len(quantiles), timesteps, features, lstm_units=features)
#     print("Model summary:", model.summary())

#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     loss_fn = QuantileLoss_2(quantiles)

#     @tf.function
#     def train_step2(x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             y_pred = model(x_batch, training=True)
#             loss = loss_fn(y_batch, y_pred)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         return loss

#     dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x)).batch(batch_size)

#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for x_batch, y_batch in dataset:
#             batch_loss = train_step2(x_batch, y_batch)
#             if tf.math.is_nan(batch_loss):
#                 print("NaN encountered in batch loss")
#                 print("NaNs in x_batch?", tf.math.reduce_any(tf.math.is_nan(x_batch)))
#                 print("NaNs in y_batch?", tf.math.reduce_any(tf.math.is_nan(y_batch)))
#             else:
#                 epoch_loss += batch_loss.numpy() / len(dataset)
        
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
#     return model
    
# def quantile_loss_3(q, y_true, y_pred):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#     error = y_true - y_pred
#     return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)

# def quantile_loss_func(quantiles):
#     def loss(y_true, y_pred):
#         losses = []
#         for i, q in enumerate(quantiles):
#             # print("shape y_pred: ", y_pred.shape)
#             # print("shape y_true: ", y_true.shape)
#             loss = quantile_loss_3(q, y_true[:, i], y_pred[:, i])
#             losses.append(loss)
#         return tf.reduce_mean(losses)
#     return loss

# def train_model_lstm(quantiles, epochs: int, lr: float, batch_size: int, x, y, n_timesteps):
#     model = QuantileRegressionLSTM(n_quantiles=len(quantiles), units=64, n_timesteps=n_timesteps)

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=quantile_loss_func(quantiles))
#     print("model summary: ", model.summary())

#     model.fit(x, y, epochs=epochs, batch_size=batch_size)

#     return model

#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     print("model summary: ", model.summary())

#     @tf.function
#     def train_step1(x_batch, y_batch):
#         with tf.GradientTape() as tape:
#             output = model(x_batch, training=True)
#             losses = []
#             for i, q in enumerate(quantiles):
#                 loss = quantile_loss_2(q, y_true=y_batch, y_pred=output[:, :, i])
#                 losses.append(loss)
#             total_loss = tf.reduce_mean(losses)
#         gradients = tape.gradient(total_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         return total_loss

#     for epoch in range(epochs):
#         dataset = tf.data.Dataset.from_tensor_slices((x, y))
#         dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size, drop_remainder=True)

#         epoch_loss = 0.0
#         j = 0
#         for x_batch, y_batch in dataset:
#             batch_loss = train_step1(x_batch, y_batch)
#             epoch_loss += batch_loss
#             j += 1
#         epoch_loss /= len(dataset)

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}, Loss: {epoch_loss.numpy()}")

#     return model
