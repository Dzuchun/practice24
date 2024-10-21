from itertools import chain

import numpy as np
import tensorflow as tf
from dataload import load_noise as lnse
from dataload import load_signal as lsig
from keras import Input, Sequential
from keras.api.optimizers import AdamW
from keras.src.layers import Dense, Dropout
from keras.src.layers.preprocessing.tf_data_layer import keras
from keras.src.metrics import AUC, BinaryAccuracy

# data stays the same in all experiments, so I'll init it once
ENTRIES = 500_000
INPUT_SIZE = 21
DROP_RATE = 0.1
L2_PAR = 0.01
VERBOSE = "auto"
entries_iter = np.fromiter(
    chain(chain(*lsig(ENTRIES)), chain(*lnse(ENTRIES))), dtype=np.float32
)
inputs_raw = tf.reshape(
    tf.convert_to_tensor(entries_iter),
    (-1, INPUT_SIZE),
)
outputs_raw = tf.concat(
    [
        tf.ones(shape=(ENTRIES,)),
        tf.zeros(shape=(ENTRIES,)),
    ],
    axis=0,
)
# shuffle for a good measure
indices = tf.random.shuffle(tf.range(start=0, limit=(2 * ENTRIES,)))
inputs = tf.gather(inputs_raw, indices, axis=0)
outputs = tf.gather(outputs_raw, indices, axis=0)

# with open("log", mode="w+") as sys.stdout:
with open("log", mode="w+") as f:
    for hidden_size in range(8, 31):
        # simple 5-layer tanh-activated model
        model = Sequential(
            [
                Input(shape=(INPUT_SIZE,)),
                Dropout(rate=DROP_RATE),
                Dense(
                    hidden_size,
                    activation=keras.activations.tanh,
                    kernel_initializer=keras.initializers.he_normal,  # type:ignore
                    # kernel_regularizer=keras.regularizers.L2(l2=L2_PAR),
                ),
                Dense(
                    hidden_size,
                    activation=keras.activations.tanh,
                    kernel_initializer=keras.initializers.he_normal,  # type:ignore
                    # kernel_regularizer=keras.regularizers.L2(l2=L2_PAR),
                ),
                Dense(
                    hidden_size,
                    activation=keras.activations.tanh,
                    kernel_initializer=keras.initializers.he_normal,  # type:ignore
                    # kernel_regularizer=keras.regularizers.L2(l2=L2_PAR),
                ),
                Dense(
                    hidden_size,
                    activation=keras.activations.tanh,
                    kernel_initializer=keras.initializers.he_normal,  # type:ignore
                    # kernel_regularizer=keras.regularizers.L2(l2=L2_PAR),
                ),
                Dense(
                    1,
                    activation=keras.activations.linear,
                    kernel_initializer=keras.initializers.he_normal,  # type:ignore
                    # kernel_regularizer=keras.regularizers.L2(l2=L2_PAR),
                ),
            ]
        )
        model.compile(
            optimizer=AdamW(),  # type:ignore
            loss=keras.losses.binary_crossentropy,
            metrics=[
                BinaryAccuracy(name="acc"),
                AUC(name="auc"),
            ],
        )

        h = model.fit(
            inputs,
            outputs,
            batch_size=100,
            epochs=10,
            validation_split=0.1,
            verbose=VERBOSE,  # type:ignore
        )
        print(
            f'For hidden size {hidden_size} got max accuracy {max(h.history["acc"])}, auc {max(h.history["auc"])}',
            file=f,
        )
        f.flush()
