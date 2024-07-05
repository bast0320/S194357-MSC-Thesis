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

# with pickly load actuals_hourly_DK1, ensembles_wind_onshore_DK1
actuals_hourly_DK1 = pd.read_pickle("loaded_variables/actuals_hourly_DK1.pkl")
ensembles_wind_onshore_DK1 = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1.pkl")
    
# Select only 10 predictors in X for now
num_of_params = 10
n_max = 1000
Y = actuals_hourly_DK1["OnshoreWindPower"][:n_max].to_numpy()
# make a index array starting from 2 to exclude date and high. prob., and ending
index = np.linspace(2, 50, num_of_params).astype(int)
X = ensembles_wind_onshore_DK1.iloc[:n_max, 2:].to_numpy() # : could be replaced by index


plt.rcParams["figure.dpi"] = int(os.environ.get("FIGURE_DPI", 150))
plt.rcParams["figure.facecolor"] = os.environ.get("FIGURE_FACECOLOR", "white")
np.random.seed(69)

import tensorflow as tf
import numpy as np

def create_data(multimodal: bool):
    x = tf.random.uniform([1000], 0.3, 10)
    y =  tf.random.gamma([1], alpha=0.1 + x / 20.0) + tf.math.log(x) # draw [shape] samples from each of the gamma distributions given, since we provide a 1000 every time...
    x = tf.random.gamma([10], alpha=0.2 + x / 50)
    
    if multimodal:
        x_extra = tf.random.uniform([500], 5, 10)
        y_extra = tf.random.normal([500], 6.0, 0.3)
        x = tf.concat([x, x_extra], axis=0)
        y = tf.concat([y, y_extra], axis=0)
    
    return  tf.expand_dims( tf.transpose(x), -1), tf.expand_dims(tf.transpose(y), -1)

multimodal: bool = False
# x, y = create_data(multimodal)
x = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(Y)
# x = tf.expand_dims(tf.convert_to_tensor(X) ,-1)
# y = tf.expand_dims(tf.convert_to_tensor(Y) ,-1)

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)
# print("x: ", x[:10])
# print("y: ", y[:10,:10])

def quantile_loss(q, y_true, y_pred):
    print("y_true: ", y_true.shape)
    print("y_pred: ", y_pred.shape)
    e = y_true - y_pred
    return tf.maximum(q * e, (q - 1.0) * e)

def calculate_error(q, y_true, y_pred):
    # y_true = tf.linspace(10.0, 20.0, 100)
    # y_pred = tf.linspace(10.0, 20.0, 200)
    loss = tf.map_fn(lambda y_pred: quantile_loss(q, y_true, y_pred), y_pred, fn_output_signature=tf.float32)
    loss = tf.reduce_mean(loss, axis=1)
    return y_true, y_pred, loss
 

q = 0.8
# y_true, y_pred, loss = calculate_error(q)
# q_true = tfp.stats.percentile(y_true, int(100*q))

# print(q_true)

# fig = plt.figure()
# plt.plot(y_pred, loss)
# plt.vlines(q_true, 0, loss.max(), linestyles="dashed", colors="k")
# plt.gca().set_xlabel("y_pred")
# plt.gca().set_ylabel("loss")
# plt.title(f"Q({q:.2f}) = {q_true:.1f}")
# plt.close()
 
import tensorflow_probability as tfp
import tensorflow as tf


class QuantileRegression(tf.keras.Model):
    def __init__(self, n_quantiles: int):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(50, activation='relu')
        self.dense3 = tf.keras.layers.Dense(n_quantiles)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

### CHATGPT VERSIONS--------------------vvvvvv
def quantile_loss(q, y_true, y_pred):
 
    # Ensure y_true is expanded to match the dimensions of y_pred
    y_true = tf.expand_dims(y_true, -1)  # This changes shape from [batch_size] to [batch_size, 1]
    e = y_true - y_pred
    return tf.maximum(q * e, (q - 1) * e)

### CHATGPT VERSIONS--------------------^^^^^^

class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        loss = tf.map_fn(lambda q: quantile_loss(q, y_true, y_pred), self.quantiles, fn_output_signature=tf.float32)
        return tf.reduce_sum(loss, axis=-1)  # sum or mean here. I think sum is correct



def train_model(quantiles, epochs: int, lr: float, eager: bool,x,y):
    model = tf.keras.Sequential([
        QuantileRegression(n_quantiles=len(quantiles))
    ])
    model.compile(
        loss=QuantileLoss(quantiles),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        run_eagerly=eager
    )
    model.fit(x, y, epochs=epochs, batch_size=64, verbose=0)
    return model

if not multimodal:
    quantiles = (0, 0.05, 0.1, 0.25, 0.28, 0.3, 0.5, 0.7, 0.9, 0.95,1)
else:
    quantiles = tf.linspace(0.05, 0.95, 9)

model = train_model(quantiles=quantiles, epochs=3000, lr=1e-4, eager=False,x=x,y=y)
model.summary()

output = model(x)
print(output.shape) # the output should be 1000,11 in shape, not a 1000,10,11


# %%
# plot the results
x = x.numpy().squeeze(-1)
y = y.numpy().squeeze(-1)
#%%
index_ = np.arange(x.shape[0]*0.1)

fig, ax = plt.subplots()
ax.plot(index_, x[:len(index_), 0], label="X", alpha=0.5)

for i in range(1, x.shape[1]):
    ax.plot(index_, x[:len(index_), i], alpha=0.1, color="blue")

ax.plot(index_, y[:len(index_)], label="Y", alpha=1, color="red", linewidth=2)
j = 0
for i, q in enumerate(quantiles):
    ax.plot(index_, output[:len(index_), i], label=f"Q({q:.2f})", alpha=0.5, color="black", linestyle="dashed", linewidth=2)

ax.legend()
plt.show()


# %%
