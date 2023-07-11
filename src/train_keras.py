# ruff: noqa: E402

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import string

import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Bidirectional, Dense, Embedding
from keras.models import Sequential

# from keras.preprocessing import sequence
from keras.utils import pad_sequences

from utils.paths import FINAL_PATH

RACES = ("asian", "black", "hispanic", "white")
RACES_DICT = {r: i for i, r in enumerate(RACES)}

# from sklearn.metrics import classification_report, confusion_matrix


# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def name2id(name: str, l: int = 15):
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

train = pd.read_parquet(FINAL_PATH / "fl_imb_deduped_train.parquet")
test = pd.read_parquet(FINAL_PATH / "ppp_test_sample.parquet")

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
feature_len = 30  # cut texts after this number of words
batch_size = 512

print("Pad sequences (samples x time)")
X_train = pad_sequences(X_train, maxlen=feature_len)
X_test = pad_sequences(X_test, maxlen=feature_len)

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

print(model.summary())

# choose between learning rates
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=(FINAL_PATH / "fl_keras2/checkpoint.h5").as_posix()
)

# train model
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[callback, checkpoint],
    class_weight={0: 3.98028843, 1: 3.80350719, 2: 1.72650826, 3: 0.3440395},
)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

model.save(FINAL_PATH / "fl_keras2/model.h5", include_optimizer=False)
model.save(FINAL_PATH / "fl_keras2/model_opt.h5")

# y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
# y_pred_bool = np.argmax(y_pred, axis=1)

# print(classification_report(np.argmax(y_test, axis=1), y_pred_bool))
# print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_bool))
