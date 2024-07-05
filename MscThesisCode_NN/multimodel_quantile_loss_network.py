import numpy as np
import matplotlib.pyplot as plt
import os
import elegy as eg
import jax
import jax.numpy as jnp
import optax

plt.rcParams["figure.dpi"] = int(os.environ.get("FIGURE_DPI", 150))
plt.rcParams["figure.facecolor"] = os.environ.get("FIGURE_FACECOLOR", "white")
np.random.seed(69)


def create_data(multimodal: bool):
    x = np.random.uniform(0.3, 10, 1000)
    y = np.log(x) + np.random.exponential(0.1 + x / 20.0)

    if multimodal:
        x = np.concatenate([x, np.random.uniform(5, 10, 500)])
        y = np.concatenate([y, np.random.normal(6.0, 0.3, 500)])

    return x[..., None], y[..., None]


multimodal: bool = False

x, y = create_data(multimodal)

# fig = plt.figure()
# plt.scatter(x[..., 0], y[..., 0], s=20, facecolors="none", edgecolors="k")
# plt.close()




def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return jnp.maximum(q * e, (q - 1.0) * e)


def calculate_error(q):
    y_true = np.linspace(10, 20, 100)
    y_pred = np.linspace(10, 20, 200)

    loss = jax.vmap(quantile_loss, in_axes=(None, None, 0))(q, y_true, y_pred)
    loss = loss.mean(axis=1)

    return y_true, y_pred, loss


q = 0.8
y_true, y_pred, loss = calculate_error(q)
q_true = np.quantile(y_true, q)


fig = plt.figure()
plt.plot(y_pred, loss)
plt.vlines(q_true, 0, loss.max(), linestyles="dashed", colors="k")
plt.gca().set_xlabel("y_pred")
plt.gca().set_ylabel("loss")
plt.title(f"Q({q:.2f}) = {q_true:.1f}")
plt.close()


'''
Generally, we would need to create to create a model per quantile. 
However, if we use a neural network, we can output the predictions 
for all the quantiles simultaneously. Here will use elegy to create
a neural network with two hidden layers with relu activations and 
linear layers with n_quantiles output units.

Think this runs on the gpu even, which is nice.

'''


class QuantileRegression(eg.Module):
    def __init__(self, n_quantiles: int):
        super().__init__()
        self.n_quantiles = n_quantiles

    def call(self, x):
        x = eg.nn.Linear(128)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(64)(x)
        x = jax.nn.relu(x)
        x = eg.nn.Linear(self.n_quantiles)(x)

        return x

class QuantileLoss(eg.Loss):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = np.array(quantiles)

    def call(self, y_true, y_pred):
        loss = jax.vmap(quantile_loss, in_axes=(0, None, -1), out_axes=1)(
            self.quantiles, y_true[:, 0], y_pred
        )
        return jnp.sum(loss, axis=-1) # sum or mean here. I think sum is correct
    

'''
Notice that we use the same quantile_loss that we created previously, along with 
some jax.vmap magic to properly vectorize the function. Finally, we will create 
a simple function that creates and trains our model for a set of quantiles using eg.
'''



def train_model(quantiles, epochs: int, lr: float, eager: bool):
    model = eg.Model(
        QuantileRegression(n_quantiles=len(quantiles)),
        loss=QuantileLoss(quantiles),
        optimizer=optax.adamw(lr),
        run_eagerly=eager,
    )

    model.fit(x, y, epochs=epochs, batch_size=64, verbose=0)

    return model


if not multimodal:
    quantiles = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
else:
    quantiles = np.linspace(0.05, 0.95, 9)

model = train_model(quantiles=quantiles, epochs=3001, lr=1e-4, eager=False)

model.summary(x)