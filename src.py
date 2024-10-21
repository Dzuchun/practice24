from dataload import PARAMS_PER_ENTRY as PPE
from dataload import load_noise as lnse
from dataload import load_signal as lsig

from itertools import chain
from types import NoneType
from typing import Callable, List, NamedTuple, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import KerasTensor, Layer, Metric, Model, Optimizer, Variable, ops
from keras.api.optimizers import RMSprop
from keras.src.layers import Dense, Dropout
from keras.src.metrics import AUC, BinaryAccuracy, Mean
from numpy._typing import NDArray
from tensorflow import Tensor, float32


class MyLayer(Layer):

    _dense: Dense
    _dropout: Dropout

    def __init__(
        self, input: Tuple[int, ...] | int, output: int, activation, *argv, **kwargs
    ):
        super().__init__(
            activity_regularizer=None,
            trainable=True,
            dtype=float32,
            *argv,
            **kwargs,
        )
        self._dense = Dense(
            output,
            activation=activation,
            input_shape=input,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.5
            ),  # type:ignore
            bias_initializer=keras.initializers.RandomUniform(
                minval=-1.0, maxval=1.0
            ),  # type:ignore
            # kernel_constraint=keras.constraints.MaxNorm(),
            # bias_constraint=keras.constraints.MaxNorm(),
        )
        self._dropout = Dropout(rate=0.003)

    def call(self, inputs, *argv, **kvargs):
        return self._dropout(self._dense(inputs, *argv, **kvargs), *argv, **kvargs)


def _layer(
    input: Tuple[int, ...] | int, output: int, activation_name: str = "tanh"
) -> Layer:
    activation = None
    match activation_name:
        case "tanh":
            activation = keras.activations.gelu
        case "lin":
            activation = keras.activations.linear
        case _:
            exit("Unknown untyped activation")
    return MyLayer(input, output, activation=activation)


def _encdec(
    input: int, intermediate: List[int], output: int
) -> Tuple[List[Layer], List[Layer]]:
    enc = [_layer(input=input, output=intermediate[0])]
    dec = [_layer(input=intermediate[0], output=input)]
    for i in range(len(intermediate) - 1):
        enc.append(_layer(input=intermediate[i], output=intermediate[i + 1]))
        dec.append(_layer(input=intermediate[i + 1], output=intermediate[i]))
    enc.append(_layer(input=intermediate[-1], output=output))
    dec.append(_layer(input=output, output=intermediate[-1]))
    return enc, dec


# tmp
ZEROES = keras.initializers.Zeros()

# metrics names
RECONSTRUCTION = "reconstruction_loss"
INT_RECONSTRUCTION = "intermediate_reconstruction_losses"
CLASSIFICATION = "classification_loss"
STABILITY = "stability_loss"


class HybridNet(Model):
    _intermediate_sizes: List[int]
    _classes: int
    _class_latent_size: int
    _detail_latent_size: int
    _stability_teacher: float
    _stability_student: float

    @property
    def intermediate_layers(self) -> int:
        return len(self._intermediate_sizes)

    def __init__(
        self,
        intermediate_sizes: List[int],
        classes: int,
        class_latent_size: int,
        detail_latent_size: int,
        stability_length: int,
        *argv,
        **kvargs,
    ) -> None:
        """
        Constructs HybridNet model. For details, look though [original paper](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_10)
        """
        Model.__init__(self, *argv, **kvargs)

        self._intermediate_sizes = intermediate_sizes
        self._classes = classes
        self._class_latent_size = class_latent_size
        self._detail_latent_size = detail_latent_size

        self._stability_student = 1.0 / stability_length
        self._stability_teacher = 1.0 - self._stability_student

    _classifier_encoder: List[Layer]
    _classifier_decoder: List[Layer]
    _classifier: Layer
    _detail_encoder: List[Layer]
    _detail_decoder: List[Layer]

    _teacher_classifier_encoder: List[Layer]
    _teacher_classifier_decoder: List[Layer]
    _teacher_detail_encoder: List[Layer]
    _teacher_detail_decoder: List[Layer]

    def build(self, input_shape):
        if len(input_shape) > 2:
            exit("This model only supports one-dimensional data")
        input_size: int = input_shape[1]

        # init encoder-decoder layers
        self._classifier_encoder, self._classifier_decoder = _encdec(
            input_size, self._intermediate_sizes, self._class_latent_size
        )
        self._detail_encoder, self._detail_decoder = _encdec(
            input_size, self._intermediate_sizes, self._detail_latent_size
        )
        # make last layers have linear activation
        self._classifier_encoder[-1].activation = keras.activations.linear
        self._classifier_decoder[0].activation = keras.activations.linear
        self._detail_encoder[-1].activation = keras.activations.linear
        self._detail_decoder[0].activation = keras.activations.linear
        # init classifier layer
        self._classifier = _layer(
            (self._class_latent_size,), self._classes, activation_name="lin"
        )
        # init teacher encdecs
        self._teacher_classifier_encoder, self._teacher_classifier_decoder = _encdec(
            input_size, self._intermediate_sizes, self._class_latent_size
        )
        self._teacher_detail_encoder, self._teacher_detail_decoder = _encdec(
            input_size, self._intermediate_sizes, self._detail_latent_size
        )
        self._teacher_classifier_encoder[-1].activation = keras.activations.linear
        self._teacher_classifier_decoder[0].activation = keras.activations.linear
        self._teacher_detail_encoder[-1].activation = keras.activations.linear
        self._teacher_detail_decoder[0].activation = keras.activations.linear
        # make all of these non-trainable
        # (teacher weights are getting updated manually)
        for l in chain(
            self._teacher_detail_encoder,
            self._teacher_detail_decoder,
            self._teacher_classifier_encoder,
            self._teacher_classifier_decoder,
        ):
            l.trainable = False

    def _int_reconstruction(
        self,
        inputs: KerasTensor,
        encoder: List[Layer],
        decoder: List[Layer],
        *argv,
        **kvargs,
    ) -> Tuple[KerasTensor, KerasTensor, KerasTensor]:
        """
        Returns a tuple of:
        - Reconstructed original input
        - Latent class layer values
        - Intermediate reconstruction losses (note: mean squared error is hard-coded here)
        """
        head: KerasTensor = inputs
        encoding_results: List[KerasTensor] = []
        for i in range(len(encoder)):
            head = encoder[i](head, *argv, **kvargs)
            encoding_results.append(head)
        encoding_results.pop()
        latent: KerasTensor = head
        losses: KerasTensor = ZEROES(shape=(0,))  # type:ignore
        for dec, enc_res in zip(
            decoder[::-1],
            encoding_results[::-1],
        ):
            # move head by one
            head = dec(head, *argv, **kvargs)
            # calculate loss
            int_loss = ops.mean(keras.losses.mean_squared_error(enc_res, head))
            # append it to result
            losses = ops.append(losses, int_loss)  # type:ignore
        # last decoder layer will be left unused:
        reconstruction = decoder[0](head)
        return reconstruction, latent, losses

    def update_teacher(self):
        def _update_teacher_inner(
            teacher: List[Layer],
            student: List[Layer],
            teacher_weight: float,
            student_weight: float,
        ):
            for tl, sl in zip(teacher, student):
                for t, s in zip(tl.weights, sl.weights):
                    t.assign(t * teacher_weight + s * student_weight)

        _update_teacher_inner(
            self._teacher_classifier_encoder,
            self._classifier_encoder,
            self._stability_teacher,
            self._stability_student,
        )
        _update_teacher_inner(
            self._teacher_classifier_decoder,
            self._classifier_decoder,
            self._stability_teacher,
            self._stability_student,
        )
        _update_teacher_inner(
            self._teacher_detail_encoder,
            self._detail_encoder,
            self._stability_teacher,
            self._stability_student,
        )
        _update_teacher_inner(
            self._teacher_detail_decoder,
            self._detail_decoder,
            self._stability_teacher,
            self._stability_student,
        )

    def _reconstruct_teacher(self, inputs) -> KerasTensor:
        rec_class, _, _ = self._int_reconstruction(
            inputs, self._teacher_classifier_encoder, self._teacher_classifier_decoder
        )
        rec_detail, _, _ = self._int_reconstruction(
            inputs, self._teacher_detail_encoder, self._teacher_detail_decoder
        )
        return ops.add(rec_class, rec_detail)  # type:ignore

    def call(
        self, inputs: keras.KerasTensor, training: bool, *argv, **kvargs
    ) -> List[KerasTensor]:
        """
        Returns:
        - predicted classes
        - class branch reconstructed input
        - detail branch reconstructed input
        - teacher reconstructed input
        - intermediate reconstruction losses
        """
        # first, let's deal with intermediate reconstruction.
        # since it's basically the same for classification and detail branch, I've
        # factored these actions out into separate function:
        rec_class, latent_class, int_reconstructiones_loss_class = (
            self._int_reconstruction(
                inputs,
                self._classifier_encoder,
                self._classifier_decoder,
                training=training,
                *argv,
                **kvargs,
            )
        )
        rec_detail, _, int_reconstruction_losses_detail = self._int_reconstruction(
            inputs,
            self._detail_encoder,
            self._detail_decoder,
            training=training,
            *argv,
            **kvargs,
        )
        int_reconstruction_losses: KerasTensor = ops.stack(
            [int_reconstructiones_loss_class, int_reconstruction_losses_detail]
        )  # type:ignore

        # predict the classes
        predicted_classes: KerasTensor = self._classifier(
            latent_class, training=training, *argv, **kvargs
        )

        # mean teacher loss
        teacher_rec: KerasTensor = self._reconstruct_teacher(inputs)

        return [
            predicted_classes,
            rec_class,
            rec_detail,
            teacher_rec,
            int_reconstruction_losses,
        ]

    def compute_output_shape(self, input_shape, *argv, **kvargs):
        return [
            (None, self._classes),
            input_shape,
            input_shape,
            input_shape,
            (2, len(self._intermediate_sizes)),
        ]


@tf.function
def classification_loss(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    return ops.mean(
        keras.losses.binary_crossentropy(true_labels, predicted_labels), axis=-1
    )  # type:ignore


@tf.function
def reconstruction_loss(
    original: Tensor,
    rec_class: Tensor,
    rec_detail: Tensor,
) -> Tensor:
    # stop the gradient to better branch
    loss_class: Tensor = ops.mean(
        keras.losses.mean_squared_error(original, rec_class)
    )  # type:ignore
    loss_detail: Tensor = ops.mean(
        keras.losses.mean_squared_error(original, rec_detail)
    )  # type:ignore
    rec_class, rec_detail = ops.cond(
        ops.greater(loss_class, loss_detail),
        lambda: (rec_class, ops.stop_gradient(rec_detail)),
        lambda: (ops.stop_gradient(rec_class), rec_detail),
    )  # type:ignore
    # calculate reconstruction and it's loss
    reconstruction: Tensor = ops.add(rec_class, rec_detail)  # type:ignore
    return ops.mean(
        keras.losses.mean_squared_error(original, reconstruction)
    )  # type:ignore


@tf.function
def stability_loss(rec_student: Tensor, rec_teacher: Tensor) -> Tensor:
    return ops.mean(
        keras.losses.mean_squared_error(rec_student, rec_teacher)
    )  # type:ignore


# Keras does not support non-scalar metrics, so here I am:
# (I copied this code from keras module. you can do literally nothing in this framework,
# and you won't even know about it until your code is executed. this is torture.)
class IntReconstruction(Metric):

    _total: Variable
    _count: Variable

    def __init__(self, shape):
        super().__init__(name="int_reconstruction", dtype=tf.float32)
        self._total = self.add_variable(
            shape=shape,
            initializer=ZEROES,
            dtype=self.dtype,
            name="total",
        )
        self._count = self.add_variable(
            shape=(),
            initializer=ZEROES,
            dtype=self.dtype,
            name="count",
        )

    def update_state(self, values: Tensor):
        if self._total.shape != values.shape:
            exit(
                f"Value update must have the same shape; present: {self._total.shape}, new: {values.shape}"
            )
        self._total.assign(self._total + values)
        self._count.assign(self._count + 1)

    def reset_state(self):
        self._total.assign(Variable(shape=self._total.shape, initializer=ZEROES))
        self._count.assign(0)

    def result(self):
        return ops.divide_no_nan(self._total, ops.cast(self._count, dtype=self.dtype))


class MetricEntry(NamedTuple):
    loss_total: float
    loss_class: float
    loss_reco: float
    loss_int_reco: NDArray
    loss_stab: float
    accuracy: float
    auc: float


class LogEntry(NamedTuple):
    epoch: int
    training: MetricEntry
    validation: MetricEntry


def train_hybrid(
    model: HybridNet,
    optimizer: Optimizer,
    train_data: Tuple[Tensor, Tensor],
    validation_data: Tuple[Tensor, Tensor],
    weights: Tuple[float, float, List[float], float],
    epochs: int,
    batch_size: int,
    logger: Callable[[LogEntry], NoneType] = lambda _: None,
):
    if len(weights[2]) != model.intermediate_layers:
        exit("Should have 1 more intermediate layer sizes than weights")
    if len(train_data[0]) % batch_size != 0:
        exit("Batch size must a divisor of train data size")
    if len(validation_data[0]) % batch_size != 0:
        exit("Batch size must a divisor of validation data size")
    int_reconstruction_weights = tf.convert_to_tensor(weights[2])

    # reshape the data appropriately
    train_batches: int = len(train_data[0]) // batch_size
    train_data = (
        tf.reshape(train_data[0], shape=(train_batches, batch_size, -1)),
        tf.reshape(train_data[1], shape=(train_batches, batch_size, -1)),
    )
    validation_batches: int = len(validation_data[0]) // batch_size
    validation_data = (
        tf.reshape(validation_data[0], shape=(validation_batches, batch_size, -1)),
        tf.reshape(validation_data[1], shape=(validation_batches, batch_size, -1)),
    )

    # init metrics
    met_loss = Mean(name="loss")
    met_classification_loss = Mean(name="classification loss")
    met_stability_loss = Mean(name="stability loss")
    met_reconstruction_loss = Mean(name="reconstruction loss")
    met_intermediate_reconstruction_losses = IntReconstruction(
        shape=(2, model.intermediate_layers)
    )
    met_accuracy = BinaryAccuracy()
    met_auc = AUC()

    # graph-compile reusable functions
    @tf.function
    def compute_loss(
        original: Tensor, classes: Tensor, training: bool, *argv, **kvargs
    ) -> Tensor:
        [
            predicted_classes,
            rec_class,
            rec_detail,
            teacher_rec,
            int_reconstruction_losses,
        ] = model(original, training=training, *argv, **kvargs)
        reconstruction: Tensor = ops.add(rec_class, rec_detail)  # type:ignore
        loss_class: Tensor = classification_loss(
            classes, predicted_classes
        )  # type:ignore
        loss_reco: Tensor = reconstruction_loss(
            original, rec_class, rec_detail
        )  # type:ignore
        loss_stab: Tensor = stability_loss(reconstruction, teacher_rec)  # type:ignore
        int_loss_weighted = tf.reduce_sum(
            tf.multiply(
                tf.reduce_sum(int_reconstruction_losses, axis=0),
                int_reconstruction_weights,
            ),
            axis=0,
        )
        loss_total: Tensor = (
            tf.multiply(loss_class, weights[0])
            + tf.multiply(loss_reco, weights[1])
            + int_loss_weighted
            + tf.multiply(loss_stab, weights[3])
        )  # type:ignore
        # update metrics
        met_loss.update_state(loss_total)
        met_classification_loss.update_state(loss_class)
        met_stability_loss.update_state(loss_stab)
        met_reconstruction_loss.update_state(loss_reco)
        met_intermediate_reconstruction_losses.update_state(int_reconstruction_losses)

        met_accuracy.update_state(classes, predicted_classes)
        met_auc.update_state(classes, predicted_classes)

        return loss_total

    @tf.function
    def train_step(original: Tensor, classes: Tensor):
        with tf.GradientTape() as tape:
            loss = compute_loss(original, classes, training=True)
        # compute a gradient and propagate changes
        gradient = tape.gradient(loss, model.trainable_weights)
        optimizer.apply(gradient, model.trainable_weights)
        # update model's teacher (part of stability loss loop)
        model.update_teacher()

    @tf.function
    def validation_step(original: Tensor, classes: Tensor):
        _ = compute_loss(original, classes, training=False)

    for epoch in range(1, epochs + 1):
        # reset metrics
        met_loss.reset_state()
        met_classification_loss.reset_state()
        met_stability_loss.reset_state()
        met_intermediate_reconstruction_losses.reset_state()
        met_accuracy.reset_state()
        met_auc.reset_state()
        # perform training
        for batch_no in range(train_batches):
            original, classes = (
                tf.reshape(
                    tf.gather(train_data[0], indices=[batch_no], axis=0),
                    shape=(batch_size, -1),
                ),
                tf.reshape(
                    tf.gather(train_data[1], indices=[batch_no], axis=0),
                    shape=(batch_size, -1),
                ),
            )
            train_step(original, classes)
        # save metrics values
        train_entry = MetricEntry(
            met_loss.result().numpy(),  # type:ignore
            met_classification_loss.result().numpy(),  # type:ignore
            met_reconstruction_loss.result().numpy(),  # type:ignore
            met_intermediate_reconstruction_losses.result().numpy(),  # type:ignore
            met_stability_loss.result().numpy(),  # type:ignore
            met_accuracy.result().numpy(),  # type:ignore
            met_auc.result().numpy(),  # type:ignore
        )
        # reuse metrics
        met_loss.reset_state()
        met_classification_loss.reset_state()
        met_stability_loss.reset_state()
        met_intermediate_reconstruction_losses.reset_state()
        met_accuracy.reset_state()
        met_auc.reset_state()
        # perform validation
        for batch_no in range(validation_batches):
            original, classes = (
                tf.reshape(
                    tf.gather(validation_data[0], indices=[batch_no], axis=0),
                    shape=(batch_size, -1),
                ),
                tf.reshape(
                    tf.gather(validation_data[1], indices=[batch_no], axis=0),
                    shape=(batch_size, -1),
                ),
            )
            validation_step(original, classes)
        # save metrics values
        validation_entry = MetricEntry(
            met_loss.result().numpy(),  # type:ignore
            met_classification_loss.result().numpy(),  # type:ignore
            met_reconstruction_loss.result().numpy(),  # type:ignore
            met_intermediate_reconstruction_losses.result().numpy(),  # type:ignore
            met_stability_loss.result().numpy(),  # type:ignore
            met_accuracy.result().numpy(),  # type:ignore
            met_auc.result().numpy(),  # type:ignore
        )
        # log metrics
        logger(LogEntry(epoch, train_entry, validation_entry))


# load data
SIGNAL_ENTRIES = 100_000
NOISE_ENTRIES = 100_000
ENTRIES = SIGNAL_ENTRIES + NOISE_ENTRIES

input_data_raw = tf.reshape(
    tf.convert_to_tensor(
        np.fromiter(
            chain(chain(*lsig(SIGNAL_ENTRIES), chain(*lnse(NOISE_ENTRIES)))),
            dtype=np.float32,
        )
    ),
    (-1, PPE),
)
label_data_raw = tf.concat(
    [tf.ones(shape=(SIGNAL_ENTRIES,)), tf.zeros(shape=(NOISE_ENTRIES,))], axis=0
)

indices = tf.random.shuffle(tf.range(start=0, limit=ENTRIES))
input_data = tf.gather(input_data_raw, indices)
label_data = tf.gather(label_data_raw, indices)

# reserve about 10% for validation
validation_entries = ENTRIES // 10
validation_input = input_data[:validation_entries, :]
validation_labels = label_data[:validation_entries]
train_input = input_data[validation_entries:, :]
train_labels = label_data[validation_entries:]

# weights:
# - classification
# - reconstruction
# - intermediate reconstruction
# - stability
loss_weights = 20.0, 10.0, [5.0, 4.0, 2.0, 1.0], 100.0
# model construction & fit
model = HybridNet(
    intermediate_sizes=[64, 32, 32, 32],
    classes=1,
    class_latent_size=10,
    detail_latent_size=3,
    stability_length=4,
    name="HybridNet",
)

optimizer = RMSprop()


def logger(entry: LogEntry):
    print(f"\n------ EPOCH {entry.epoch:=3} ------")
    print(
        f"Total loss: {entry.validation.loss_total:.3f} ({entry.training.loss_total:.3f} training)"
    )
    print(
        f"Classification loss: {entry.validation.loss_class:.3f} ({entry.training.loss_class:.3f} training)"
    )
    print(
        f"Reconstruction loss: {entry.validation.loss_reco:.3f} ({entry.training.loss_reco:.3f} training)"
    )
    print(
        f"Stability loss: {entry.validation.loss_stab:.3f} ({entry.training.loss_stab:.3f} training)"
    )
    print(
        f"Accuracy: {entry.validation.accuracy:.3f} ({entry.training.accuracy:.3f} training)"
    )
    print(f"AUC: {entry.validation.auc:.3f} ({entry.training.auc:.3f} training)")
    print(f"Intermediate reconstruction loss:\n{entry.validation.loss_int_reco}")
    print(f"(training:)\n{entry.training.loss_int_reco}")
    pass


train_hybrid(
    model=model,
    optimizer=optimizer,
    train_data=(train_input, train_labels),
    validation_data=(validation_input, validation_labels),
    weights=loss_weights,
    epochs=1000,
    batch_size=500,
    logger=logger,
)
