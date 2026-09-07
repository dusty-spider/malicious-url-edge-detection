import tensorflow as tf
import numpy as np

models_to_prune = [
    {"model_path": "models/cnn_lstm1.keras",    "output_path": "models/cnn_lstm_pruned70.keras"},
    {"model_path": "models/baseline_snn.h5",    "output_path": "models/baseline_cnn_pruned70.h5"},
]

EPOCHS = 8 
BATCH_SIZE = 512


def prune_model(model_path, output_path):
    model = tf.keras.models.load_model(model_path)
# old thing: model = tf.keras.models.load_model("models/cnn_lstm1.keras")

    total_params = 0
    zero_params = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            new_weights = []
            for i, w in enumerate(weights):
                # skip biases (1D arrays) - only prune kernel weights or else they lose the accuracy
                if w.ndim == 1:
                    new_weights.append(w)
                    continue
                threshold = np.percentile(np.abs(w), 70) #SPARSITY here = 70%
                mask = np.abs(w) >= threshold
                pruned_w = w * mask
                total_params += w.size
                zero_params += np.sum(pruned_w == 0)
                new_weights.append(pruned_w)
            layer.set_weights(new_weights)

    print(f"Sparsity: {zero_params / total_params:.2%}")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    X_train = np.load("X_train1.npy")
    y_train = np.load("y_train1.npy")
    X_val = np.load("X_val1.npy")
    y_val = np.load("y_val1.npy")

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    model.save(output_path)

def main():
    for x in models_to_prune:
        prune_model(**x)


#at first tried 50% sparcity with 5 epochs but result were so limited i switched to 70 and 8 epochs
if __name__ == "__main__":
    main()