from collections.abc import Callable
from math import cos, sin, tau
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model, Optimizer
from keras.src.layers import Dense
from keras.src.layers.merging.concatenate import Concatenate
from keras.src.optimizers import Adam

# the equation is as follows: `u_xx + u_yy = D*f(x, y)`
# - where f(x, y) = 1 inside square [0;1]x[0;1]
# - 0 otherwise
# boundary conditions:
# - u(0, 0) = 0
# - u(4*cos(phi), 4*sin(phi))_phi = 0

# I suspect that model must be a bit more complex for this task
input = Input(shape=(2,))
first = Dense(128, activation=tf.nn.tanh)(input)
sharp = Dense(16, activation=tf.nn.relu)(first)
second = Dense(128, activation=tf.nn.tanh)(Concatenate()([first, sharp]))
result = Dense(1)(Concatenate()([second, sharp]))
u = Model(input, result)

D = 1.0


def f(r_inside: tf.Tensor, r_outside: tf.Tensor) -> tf.Tensor:
    in_t = tf.constant(D, shape=r_inside.shape)
    out_t = tf.constant(0.0, shape=r_outside.shape)
    return tf.concat([in_t, out_t], axis=0)  # type: ignore


CENTER = tf.constant([[-1.0, 0.0]])


def loss(
    m: Model,
    r_inside: tf.Tensor,
    r_outside: tf.Tensor,
    r_boundary: tf.Tensor,
    f: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> Tuple[float, float, float, float]:
    # first, let's calculate the Laplasian
    # it can be found as a trace of Hessian matrix:
    r: tf.Tensor = tf.concat([r_inside, r_outside], axis=0)  # type: ignore
    with tf.GradientTape() as tp1:
        tp1.watch(r)
        with tf.GradientTape() as tp2:
            tp2.watch(r)
            u = m(r)
        u_grad: tf.Tensor = tp2.batch_jacobian(u, r)  # type: ignore
    u_hess: tf.Tensor = tp1.batch_jacobian(u_grad, r)  # type:ignore
    # print(u_hess.shape)
    laplace = tf.linalg.trace(u_hess)
    # so here comes the equation loss:
    eq_loss = tf.reduce_mean(tf.square(tf.subtract(laplace, f(r_inside, r_outside))))

    # next is boundary condition. for that, we will compute the gradient and dot-multiply
    # it with rotated by t/4 norm vectors (which happen to be normalized `r` in this case)
    with tf.GradientTape() as tp:
        tp.watch(r_boundary)
        u2 = m(r_boundary)
    u_grad1 = tp.gradient(u2, r_boundary)
    r_norm = tf.divide(r_boundary, tf.norm(r_boundary, axis=1, keepdims=True))
    # a rotation of normed (x, y) by t/4 is just (y, -x)
    r_x, r_y = tf.split(r_norm, axis=1, num_or_size_splits=2)  # type:ignore
    r_norm_rot = tf.concat([r_y, tf.negative(r_x)], axis=1)
    u_r_tan = tf.reduce_prod([u_grad1, r_norm_rot], axis=0)
    bc_loss = tf.reduce_mean(tf.square(u_r_tan))

    # finally, a center loss, so that function could find it's rest point
    u_ins = m(r_inside)
    sp_loss = tf.square(m(CENTER)) + tf.reduce_mean(
        tf.square(tf.subtract(u_ins, tf.constant(-1.0)))
    )
    return eq_loss + bc_loss + sp_loss, eq_loss, bc_loss, sp_loss


def train_step(
    m: Model,
    opt: Optimizer,
    r_inside: tf.Tensor,
    r_outside: tf.Tensor,
    r_boundary: tf.Tensor,
    f: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> Tuple[float, float, float, float]:
    with tf.GradientTape() as tp:
        total_loss, eq_loss, bc_loss, sp_loss = loss(
            m, r_inside, r_outside, r_boundary, f
        )
    vars = m.trainable_variables
    grad = tp.gradient(total_loss, vars)
    opt.apply_gradients(zip(grad, vars))  # type: ignore
    return float(total_loss), float(eq_loss), float(bc_loss), float(sp_loss)


# points for training
INSIDE_POINTS = 100
OUTSIDE_POINTS = 1500
BOUNDARY_POINTS = 500
BOUNDARY_RADII = 4.0
inside_points = tf.convert_to_tensor(
    np.random.uniform(0.0, 1.0, size=(INSIDE_POINTS, 2)), dtype=tf.float32
)
outside_points = []
while len(outside_points) < OUTSIDE_POINTS:
    x = np.random.uniform(-BOUNDARY_RADII, BOUNDARY_RADII)
    y = np.random.uniform(-BOUNDARY_RADII, BOUNDARY_RADII)
    if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
        # this point is inside the source
        continue
    if np.hypot(x, y) >= BOUNDARY_RADII:
        # this point is outside of the domain
        continue
    outside_points.append([x, y])
outside_points = tf.convert_to_tensor(outside_points, dtype=tf.float32)
boundary_points = tf.convert_to_tensor(
    [
        [BOUNDARY_RADII * cos(phi), BOUNDARY_RADII * sin(phi)]
        for phi in np.random.uniform(0.0, tau, size=(BOUNDARY_POINTS,))
    ],
    dtype=tf.float32,
)

# training
epoch_iterations = 10
epochs = 100
adam = Adam()
for epoch in range(epochs):
    for _ in range(epoch_iterations - 1):
        train_step(u, adam, inside_points, outside_points, boundary_points, f)
    l = train_step(u, adam, inside_points, outside_points, boundary_points, f)
    print(f"Epoch {epoch}: losses = {l}")

# show the result
IMAGE_SIZE = 100
X = np.linspace(-4.0, 4.0, IMAGE_SIZE)
Y = np.linspace(-4.0, 4.0, IMAGE_SIZE)
Z = np.array([[x, y] for x in X for y in Y])
z = u(tf.convert_to_tensor(Z)).numpy().reshape((IMAGE_SIZE, IMAGE_SIZE))

plt.imshow(z, interpolation="bilinear", cmap="plasma")
plt.show()
