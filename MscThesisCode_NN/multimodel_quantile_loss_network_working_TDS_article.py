# %%
import numpy as np
import matplotlib.pyplot as plt
import os
#import elegy as eg
import jax
import jax.numpy as jnp
#import optax

import pickle
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
from FFNN_as_basis_to_TAQR import * # Standard FFNN Class, and a function "train_ffnn" to train the FFNN.
from functions_for_TAQR import * # Jan's algo in Python, currently in a very raw state 4/4-24. 
from helper_functions import * # Helper functions: "generate_spline_matrices"

# load_variables()

# do not show warnings
import warnings
warnings.filterwarnings("ignore")

# ignore tensorflow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# with pickly load actuals_hourly_DK1, ensembles_wind_onshore_DK1
actuals_hourly_DK1 = pd.read_pickle("loaded_variables/actuals_hourly_DK1.pkl")
ensembles_wind_onshore_DK1 = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1.pkl")
    
# Select only 10 predictors in X for now
num_of_params = 10

n_max = 2000 ################# INPUT: number of samples

Y = actuals_hourly_DK1["OnshoreWindPower"][:n_max].to_numpy()
# make a index array starting from 2 to exclude date and high. prob., and ending
index = np.linspace(2, 50, num_of_params).astype(int)
X = ensembles_wind_onshore_DK1.iloc[:n_max, :].to_numpy() # : could be replaced by index

X_y = np.concatenate((X, Y.reshape(-1,1)), axis=1)




plt.rcParams["figure.dpi"] = int(os.environ.get("FIGURE_DPI", 150))
plt.rcParams["figure.facecolor"] = os.environ.get("FIGURE_FACECOLOR", "white")
np.random.seed(69)

import tensorflow as tf
import numpy as np

multimodal: bool = False
# x, y = create_data(multimodal)
x = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(Y)
# x = tf.expand_dims(tf.convert_to_tensor(X) ,-1)
# y = tf.expand_dims(tf.convert_to_tensor(Y) ,-1)

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)
 
# %%

def calculate_error(q, y_true, y_pred):
    # y_true = tf.linspace(10.0, 20.0, 100)
    # y_pred = tf.linspace(10.0, 20.0, 200)
    loss = tf.map_fn(lambda y_pred: quantile_loss_2(q, y_true, y_pred), y_pred, fn_output_signature=tf.float32)
    loss = tf.reduce_mean(loss, axis=1)
    return y_true, y_pred, loss
 

q = 0.8

 
import tensorflow_probability as tfp
import tensorflow as tf


    
class QuantileRegression(tf.keras.Model):
    def __init__(self, n_quantiles: int):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(50, activation='relu')
        self.quantile_outputs = []
        for i in range(n_quantiles):
            self.quantile_outputs.append(tf.keras.layers.Dense(1)) # here depends on how much we want out per quantile...

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        quantile_outputs = []
        for output_layer in self.quantile_outputs:
            quantile_outputs.append(output_layer(x))
        return tf.concat(quantile_outputs, axis=1)

 

### CLAUD VERSIONS--------------------vvvvvv
def quantile_loss_2(q, y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # y_true = tf.expand_dims(y_true, -1)  # This changes shape from [batch_size] to [batch_size, 1]

    # 

    #print("shapes: ", y_true.shape, y_pred.shape)
    y_true = tfp.stats.percentile(y_true, 100*q, axis = 1)
    #print(y_true)

    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)

### CLAUD VERSIONS--------------------^^^^^^

class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        loss = tf.map_fn(lambda q: quantile_loss_2(q, y_true, y_pred), self.quantiles, fn_output_signature=tf.float32)
        return tf.reduce_sum(loss, axis=-1)  # sum or mean here. I think sum is correct

def train_model(quantiles, epochs: int, lr: float, batch_size: int, x, y):
    model = QuantileRegression(n_quantiles=len(quantiles))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            output = model(x_batch, training=True)
            losses = []
            for i, q in enumerate(quantiles):
                loss = quantile_loss_2(q, y_true = y_batch, y_pred = output[:, i]) # tfp.stats.percentile(y_batch,100*q)
                losses.append(loss)
            total_loss = tf.reduce_mean(losses)
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss
    
    for epoch in range(epochs):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size, drop_remainder=True)
        # dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size).repeat()
        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # dataset = tf.data.Dataset.padded_batch(dataset, batch_size=50)
        epoch_loss = 0.0
       
        j = 0
        for x_batch, y_batch in dataset:
            #print("shapes: ", x_batch.shape, y_batch.shape, "epoch: ", epoch, "step: ", j)
            batch_loss = train_step(x_batch, y_batch)
            epoch_loss += batch_loss
            j +=1
        
        
        epoch_loss /= len(dataset)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss.numpy()}")
    
    return model


quantiles = (0, 0.05, 0.1, 0.25, 0.28, 0.3, 0.5, 0.7, 0.9, 0.95,1)
quantiles = np.linspace(0.01, 0.99, 50)
steps_per_epoch = len(x) // 50

model = train_model(quantiles=quantiles, epochs=500, lr=1e-3, batch_size=50, x=x, y=X_y)
model.summary()

output = model(x)
print("output shape: ", output.shape) 


# %%
# plot the results
# x = x.numpy().squeeze(-1)
# y = y.numpy().squeeze(-1)



index_ = np.arange(x.shape[0]* 0.2)

fig, ax = plt.subplots()
ax.plot(index_, x[:len(index_), 0], label="X", alpha=0.5)

for i in range(1, x.shape[1]):
    ax.plot(index_, x[:len(index_), i], alpha=0.05, color="blue")

ax.plot(index_, output[:len(index_), 0], label="Quantiles", alpha=0.25, color="black", linestyle="dashed", linewidth=2)

for i, q in enumerate(quantiles[1:]):
    ax.plot(index_, output[:len(index_), i],  alpha=0.25, color="black", linestyle="dashed", linewidth=2) # label=f"Q({q:.2f})",

ax.plot(index_, y[:len(index_)], label="Y", alpha=1, color="red", linewidth=2)
ax.legend()
plt.show()


# %%
# now lets run the trained model on the test data, 200 samples
steps = 1000
Y_test = actuals_hourly_DK1["OnshoreWindPower"][n_max:(n_max+steps)].to_numpy()
X_test = ensembles_wind_onshore_DK1.iloc[n_max:(n_max+steps), :].to_numpy() # : could be replaced by index

X_y_test = np.concatenate((X_test, Y_test.reshape(-1,1)), axis=1)

x_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(Y_test)

output_test = model(x_test)

index_ = np.arange(x_test.shape[0] )

fig, ax = plt.subplots()
ax.plot(index_, x_test[:len(index_), 0], label="X", alpha=0.5)

for i in range(1, x_test.shape[1]):
    ax.plot(index_, x_test[:len(index_), i], alpha=0.05, color="blue")

ax.plot(index_, output_test[:len(index_), 0], label="Quantiles", alpha=0.25, color="black", linestyle="dashed", linewidth=2)

for i, q in enumerate(quantiles[1:]):
    ax.plot(index_, output_test[:len(index_), i],  alpha=0.25, color="black", linestyle="dashed", linewidth=2) # label=f"Q({q:.2f})",

ax.plot(index_, y_test[:len(index_)], label="Y", alpha=1, color="red", linewidth=2)
ax.legend()
title = "Test data"
ax.set_title(title)
plt.show()

# save the data to loaded_variables/ensembles_wind_onshore_DK1_post_NN.pkl
# save output_test as pickle

with open("loaded_variables/ensembles_wind_onshore_DK1_post_NN_index_2_to_3k.pkl", "wb") as f:
    pickle.dump(output_test, f)



# %%
# Save the weights
model.save('saved_models/50_quantiles_18_apr_2024.keras')
# %%
# # Create a new model instance
# model = create_model()

# # Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')





# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
import pandas as pd
from FFNN_as_basis_to_TAQR import * # Standard FFNN Class, and a function "train_ffnn" to train the FFNN.

from functions_for_TAQR import * # Jan's algo in Python, currently in a very raw state 4/4-24. 

from helper_functions import * # Helper functions: "generate_spline_matrices"

# 22. april, running TAQR with 
# with pickly load actuals_hourly_DK1, ensembles_wind_onshore_DK1
actuals_hourly_DK1 = pd.read_pickle("loaded_variables/actuals_hourly_DK1.pkl")
ensembles_wind_onshore_DK1 = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1.pkl")
    
ensembles_wind_onshore_DK1_corrected_by_NN = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1_post_NN_index_2_to_3k.pkl")
# Select only 10 predictors in X for now
num_of_params = 10

n_max = 2000 ################# INPUT: number of samples
steps = 1000


Y_test = actuals_hourly_DK1["OnshoreWindPower"][n_max:(n_max+steps)].to_numpy()
X_test_1 = ensembles_wind_onshore_DK1.iloc[n_max:(n_max+steps), :].to_numpy()

X_test_2 = ensembles_wind_onshore_DK1_corrected_by_NN[:steps, :]
# convert tensor to numpy
X_test_2 = X_test_2.numpy() 

n_init = 350
n_full = 1000

# clean for nans
Y_test[np.isnan(Y_test)] = 0

print("Y_test shape: ", Y_test.shape)
print("X_test_1 shape: ", X_test_1.shape)
print("X_test_2 shape: ", X_test_2.shape)

y_pred, y_true, BETA = one_step_quantile_prediction(X_test_1, Y_test, n_init, n_full, quantile = 0.1)


# we now want to wrap this for X_test_1 and 2, and for 5 quantiles for each...
y_pred_1 = []
y_true_1 = []
BETA_1 = []
y_pred_2 = []
y_true_2 = []
BETA_2 = []

quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
for q in quantiles:
    y_pred_1_, y_true_1_, BETA_1_ = one_step_quantile_prediction(X_test_1, Y_test, n_init, n_full, quantile = q)
    y_pred_2_, y_true_2_, BETA_2_ = one_step_quantile_prediction(X_test_2, Y_test, n_init, n_full, quantile = q)
    y_pred_1.append(y_pred_1_)
    y_true_1.append(y_true_1_)
    BETA_1.append(BETA_1_)
    y_pred_2.append(y_pred_2_)
    y_true_2.append(y_true_2_)
    BETA_2.append(BETA_2_)




# %%
# plot the results
index_ = np.arange(y_true.shape[0])

plt.figure()
plt.plot(index_, y_true_1[0], label="True", color="red")
for i, q in enumerate(quantiles):
    plt.plot(index_, y_pred_1[i], label=f"Predicted {q}", color="blue", alpha=0.2, linestyle="dashed")

plt.legend()
plt.title("TAQR with original ensembles")
plt.savefig("figures/TAQR_comparison_with_NN_1.pdf")
plt.show()


plt.figure()
plt.plot(index_, y_true_2[0], label="True", color="red")
for i, q in enumerate(quantiles):
    plt.plot(index_, y_pred_2[i], label=f"Predicted {q}", color="blue", alpha=0.2, linestyle="dashed")

plt.legend()
plt.title("TAQR with corrected ensembles")
plt.savefig("figures/TAQR_comparison_with_NN_2_corrected.pdf")
plt.show()


# Now we just need to quantify this with a score, and we are done!
# Variogram, CRPS, or quantile skill score... above and etc..., didn't we make a function for this?

# %%

actuals_hourly_DK1 = pd.read_pickle("loaded_variables/actuals_hourly_DK1.pkl")
ensembles_wind_onshore_DK1 = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1.pkl")
    
ensembles_wind_onshore_DK1_corrected_by_NN = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1_post_NN_index_2_to_3k.pkl")
# Select only 10 predictors in X for now
num_of_params = 10

n_max = 2000 ################# INPUT: number of samples
steps = 1000


Y_test = actuals_hourly_DK1["OnshoreWindPower"][n_max:(n_max+steps)].to_numpy()
X_test_1 = ensembles_wind_onshore_DK1.iloc[n_max:(n_max+steps), :].to_numpy()

X_test_2 = ensembles_wind_onshore_DK1_corrected_by_NN[:steps, :]
# convert tensor to numpy
X_test_2 = X_test_2.numpy() 

n_init = 350
n_full = 1000

# clean for nans
Y_test[np.isnan(Y_test)] = 0


# I need to rank the updated ensembles vs the original ensembles with the crps score, and the variogram score
# first focus on the crps score, then the variogram score

# crps

import numpy as np
import properscoring as ps

crps_1 = ps.crps_ensemble(Y_test, X_test_1)
crps_2 = ps.crps_ensemble(Y_test, X_test_2)

skill_score = 1- (np.mean(crps_1) - np.mean(crps_2)) / np.mean(crps_1)

if False:
    plt.figure()
    plt.plot(crps_1, color="blue", alpha=0.5, linewidth=2, label=f"Original ensembles, mean CRPS: {np.mean(crps_1):.2f}")
    plt.plot(crps_2, color="red", alpha=0.5, linewidth=1, label=f"Corrected ensembles, mean CRPS: {np.mean(crps_2):.2f}")
    plt.legend()
    plt.title(f"CRPS score for original and corrected ensembles, with skill score: {skill_score:.2f}")
    # plt subtitle: Remember, here we have only trained simple FFNN
    plt.suptitle("Remember, here we have only trained simple FFNN")
    plt.savefig("figures/CRPS_comparison_with_NN.pdf")
    plt.show()


# variogram
# calculate the variogram score

def variogram_score_R(x, y, p=0.5):
    """
    Calculate the Variogram score for a given quantile.
    From the paper Energy and AI, Mathias
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0

    # Iterate through all pairs
    for i in range(k - 1):
        for j in range(i + 1, k):
            Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j])**p)
            score += (np.abs(y[i] - y[j])**p - Ediff)**2

    # Variogram score
    return score


print(f"{variogram_score_R(X_test_1, Y_test, p=0.5)/len(Y_test):.2f}") # >> 409732.91
print(f"{variogram_score_R(X_test_2, Y_test, p=0.5)/len(Y_test):.2f}") # >> 352279.82

# The variogram score is also better. How do we standardise this?




# %%
# Calculate the quantile skill score

import numpy as np

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

import numpy as np

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
    assert len(y_true) == len(y_pred[0]), "y_true and y_pred must be the same length"
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

# Calculate the QSS for the original and corrected ensembles
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
QSS_1 = multi_quantile_skill_score(y_true_1[0], y_pred_1, quantiles)
QSS_2 = multi_quantile_skill_score(y_true_2[0], y_pred_2, quantiles)

print(f"QSS for original ensembles: {np.mean(QSS_1):.4f}")
print(f"QSS for corrected ensembles: {np.mean(QSS_2):.4f}")
# %%
# TODO
# - so far results are looking very promising for the corrected ensembles with the FFNN and TAQR. 
# I would like to be able to run this for all data sources, and maybe even more data sources
# So, we need to set up some kind of pipeline for this, and then we can start to look at the results

# Pipeline:
# - Load the data, based on string input => DATA
# - With the ensembles, true data, we need to train the network on 80% of the data => TRAINED MODEL
# - We then out of sample use the trained model to correct the ensembles => CORRECTED ENSEMBLES
# - We then use the corrected ensembles to run the TAQR algo => TAQR RESULTS for e.g. 5 quantiles
# - We then calculate the QSS, CRPS, and Variogram score for the TAQR results => SCORES
# _ We could consider also just calculating scores for the corrected ensembles...
# - We then save the scores, and the corrected ensembles, and the TAQR results, and the TRAINED MODEL
# - We then repeat this for all data sources, and then we can start to look at the results


# %%
