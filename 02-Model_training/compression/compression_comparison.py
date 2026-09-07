import tensorflow as tf
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score


# models load 
MODELS_TO_COMPRESS = [
    ("models/cnn_lstm1.keras",          "models/tflite_lstm"),
    ("models/baseline_cnn.h5",          "models/tflite_cnn"),
    ("models/cnn_lstm_pruned70.keras",  "models/tflitelstm_prune70"),
    ("models/baseline_cnn_pruned70.h5", "models/tflitecnn_prune70"),
]

def export(MODEL_PATH, SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    # 1. float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(f"{SAVE_DIR}/fp32.tflite", "wb").write(tflite_model)

    # 2. dynamic quant
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(f"{SAVE_DIR}/dynamic.tflite", "wb").write(tflite_model)

    # 3. float16
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open(f"{SAVE_DIR}/fp16.tflite", "wb").write(tflite_model)

    print("All models exported!!!!")


#evaluation funcion
def evaluate(model_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    start = time.time()

    for x in X_test:
        x = np.expand_dims(x, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        preds.append(pred[0][0] > 0.5)

    latency = (time.time() - start) / len(X_test)
    acc = accuracy_score(y_test, preds)

    return acc, latency


def main():
    # load test dat
    X_test = np.load("X_test1.npy")
    y_test = np.load("y_test1.npy")

    #export
    for MODEL_PATH, SAVE_DIR in MODELS_TO_COMPRESS:
        export(MODEL_PATH, SAVE_DIR)
    
    #result show off
    print("\nresults:\n")
    for MODEL_PATH, SAVE_DIR in MODELS_TO_COMPRESS:
        for variant in ["fp32", "dynamic", "fp16"]:
            path = f"{SAVE_DIR}/{variant}.tflite"
            acc, lat = evaluate(path, X_test, y_test)
            size_kb = os.path.getsize(path) / 1024
            print(f"{SAVE_DIR}/{variant}")
            print(f" Accuracy: {acc:.4f}")
            print(f" Latency: {lat*1000:.2f} ms")
            print(f" Size: {size_kb:.2f} KB\n")
        
if __name__ == "__main__":
    main()