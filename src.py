from math import pi
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model, Optimizer, Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from matplotlib import colormaps
from numpy._typing import ArrayLike, NDArray
from numpy.core.multiarray import dtype

# model to approx a function
# (seemingly can be anything, provided it's complex enough?)


u = Sequential(
    [
        Dense(50, activation="tanh", input_shape=(1,)),
        Dense(50, activation="tanh"),
        Dense(1),
    ]
)

# let's solve equation u'' = f(x)
# here's an `f(x)`, then:


def f(x: tf.Tensor) -> tf.Tensor:
    pi_t = tf.constant(pi, shape=x.shape)
    pi_sq_t = tf.square(pi_t)
    return -pi_sq_t * tf.sin(tf.multiply(pi_t, x))


def loss(m: Model, x: tf.Tensor, f: Callable[[tf.Tensor], tf.Tensor]) -> float:
    """
    Defines loss for nn-based DE+BC solution.
    - DE if of a form u''=f(x)
    - BC here is u(0)=u(1)=0
    """
    # first, we compute 2nd derivative using tensorflow's gradient tape:
    with tf.GradientTape() as tp1:
        tp1.watch(x)
        with tf.GradientTape() as tp2:
            tp2.watch(x)
            u = m(x)
        u_x = tp2.gradient(u, x)
    u_xx = tp1.gradient(u_x, x)
    # equation is u'' = f(x), so the equation loss would be
    eq_loss = tf.reduce_mean(tf.square(tf.subtract(u_xx, f(x))))

    # boundary conditions loss
    boundary = tf.constant([0.0, 1.0])
    boundary_value = m(boundary)
    bc_loss = tf.reduce_mean(tf.square(boundary_value))

    return eq_loss + bc_loss


# there's an api to do that automatically,
# but I indeed think that it's unnecessary, and had too many parts moving under the hood.
def train_step(
    m: Model, opt: Optimizer, x: tf.Tensor, f: Callable[[tf.Tensor], tf.Tensor]
) -> float:
    with tf.GradientTape() as tp:
        l = loss(m, x, f)
    vars = m.trainable_variables
    grad = tp.gradient(l, vars)
    opt.apply_gradients(zip(grad, vars))  # type: ignore
    return l


# training requires some points to be scattered around the domain. so let's do just that.
train_points = 20
x_train = np.linspace(0.0, 1.0, train_points).reshape(-1, 1).astype(np.float32)
x_train_t = tf.convert_to_tensor(x_train, dtype=tf.float32)

# we will record some data for further plotting at the same time
plot_points = 1000
x_plot = np.linspace(0.0, 1.0, plot_points).reshape(-1, 1).astype(np.float32)
x_plot_t = tf.convert_to_tensor(x_plot, dtype=tf.float32)
cmap = colormaps["plasma"]


# training itself
epoch_iterations = 100
epochs = 10
adam = Adam()
for epoch in range(epochs):
    for _ in range(epoch_iterations - 1):
        train_step(u, adam, x_train_t, f)
    l = train_step(u, adam, x_train_t, f)
    print(f"Epoch {epoch}: loss = {l}")

    # add a snapshot for plotting
    plt.plot(x_plot, u(x_plot), color=cmap(1 - epoch / (epochs - 1)))

plt.show()
