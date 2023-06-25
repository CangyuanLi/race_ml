# ruff: noqa: E402

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import string

import keras.layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Bidirectional, CuDNNLSTM, Dense, Embedding
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix

from models import FLZEmbedBiLSTM
from utils.constants import DEVICE, RACES_DICT
from utils.paths import FINAL_PATH

print(DEVICE)

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def name2id(name, l=10):
    ids = [0] * l
    for i, c in enumerate(name):
        if i < l:
            if c.isalpha():
                ids[i] = char2id.get(c, char2id["U"])
            elif c in string.punctuation:
                ids[i] = char2id.get(c, char2id[" "])
            else:
                ids[i] = char2id.get(c, char2id["U"])
    return ids


# create ASCII dictionary
chars = ["E"] + [chr(i) for i in range(97, 123)] + [" ", "U"]
id2char = {i: j for i, j in enumerate(chars)}
char2id = {j: i for i, j in enumerate(chars)}


train = pd.read_parquet(FINAL_PATH / "fl_train.parquet").sample(1000)
test = pd.read_parquet(FINAL_PATH / "fl_test.parquet").sample(1000)

X_train = [
    name2id(fn.lower()) + name2id(ln.lower())
    for fn, ln in zip(train["first_name"], train["last_name"])
]
y_train = [RACES_DICT[r] for r in train["race_ethnicity"].tolist()]

X_test = [
    name2id(fn.lower()) + name2id(ln.lower())
    for fn, ln in zip(test["first_name"], test["last_name"])
]
y_test = [RACES_DICT[r] for r in test["race_ethnicity"].tolist()]


num_words = len(id2char)
feature_len = 20  # cut texts after this number of words (among top max_features most common words)
batch_size = 256

print("Pad sequences (samples x time)")
X_train = sequence.data_utils.pad_sequences(X_train, maxlen=feature_len)
X_test = sequence.data_utils.pad_sequences(X_test, maxlen=feature_len)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

num_classes = 4
print(num_classes, "classes")

print(
    "Convert class vector to binary class matrix "
    "(for use with categorical_crossentropy)"
)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model = Sequential()
model.add(Embedding(num_words, 256, input_length=feature_len))
# try out bi-directional LSTM
model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(512, dropout=0.2)))
model.add(Dense(num_classes, activation="softmax"))


FLZEmbedBiLSTM()


class FLZEmbedBiLSTMKeras:
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0,
        num_layers: int = 1,
    ) -> None:
        if DEVICE == "cpu":
            lstm = LSTM
        else:
            lstm = CuDNNLSTM

        model = Sequential()
        model.add(Embedding(input_size, embedding_dim))
        for _ in num_layers:
            model.add(
                Bidirectional(lstm(hidden_size, return_sequences=True, dropout=dropout))
            )
        model.add(Embedding(input_size, embedding_dim))
        for _ in num_layers:
            model.add(
                Bidirectional(lstm(hidden_size, return_sequences=True, dropout=dropout))
            )
        model.add(Dense(output_size, activation="softmax"))


def build_model_lstm_zipcode_composition(
    num_words, num_classes, feature_len, output_length, num_length
):
    nlp_input = keras.layers.Input(shape=(feature_len,))
    meta_input = keras.layers.Input(shape=(num_length,))
    emb = Embedding(num_words, output_length, input_length=feature_len)(nlp_input)
    nlp_out = LSTM(100)(emb)
    concat = keras.layers.concatenate([nlp_out, meta_input])
    classifier = Dense(32, activation="relu")(concat)
    output = Dense(num_classes, activation="sigmoid")(classifier)
    model = Model(inputs=[nlp_input, meta_input], outputs=[output])
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    print(model.summary())

    return model


# FLZEmbedBiLSTMKeras = Sequential()
# FLZEmbedBiLSTMKeras.add(Embedding(num_words, 256, input_length=feature_len))
# FLZEmbedBiLSTMKeras.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.2)))


print(model.summary())

# choose between learning rates
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

# train model
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=10,
    validation_split=0.2,
    verbose=1,
    callbacks=[callback],
)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)


y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(np.argmax(y_test, axis=1), y_pred_bool))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_bool))

model.save(FINAL_PATH / "fl_keras/model.h5", include_optimizer=False)
model.save(FINAL_PATH / "fl_keras/model_opt.h5")
