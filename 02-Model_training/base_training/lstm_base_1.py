import numpy as np

# load training stuff arrays and labels
X_train = np.load("X_train1.npy")
X_val   = np.load("X_val1.npy")
X_test  = np.load("X_test1.npy")

y_train = np.load("y_train1.npy")
y_val   = np.load("y_val1.npy")
y_test  = np.load("y_test1.npy")
print(X_train.shape, y_train.shape)

#balancing the data
#importing helper from scikit to make it balanced
from sklearn.utils.class_weight import compute_class_weight

#penaltise the class weights malicious urls penalised more heavily
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
# dictionary import for keras
class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}
print(class_weight_dict)

from keras.models import Model
from keras.layers import (
    Input, Embedding, Conv1D, LSTM,
    GlobalMaxPooling1D, Dense, Dropout
)
from keras.optimizers import Adam

VOCAB_SIZE = int(np.max(X_train)) + 1
EMBED_DIM = 128
MAX_LEN = 200

inputs = Input(shape=(MAX_LEN,))

embedding = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBED_DIM
)(inputs)

# CNN feature extraction
conv = Conv1D(128, kernel_size=5, activation="relu", padding="same")(embedding)

# LSTM for sequence modelling
lstm = LSTM(64, return_sequences=True, unroll=True)(conv)
#Unroll=true was added for the model to be exportable in ftlite, making it static

# pool after LSTM
pool = GlobalMaxPooling1D()(lstm)
dense = Dense(128, activation="relu")(pool)
dropout = Dropout(0.5)(dense)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs, output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("Models/base_models/cnn_lstm1.keras", save_best_only=True)
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=512,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
#evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
