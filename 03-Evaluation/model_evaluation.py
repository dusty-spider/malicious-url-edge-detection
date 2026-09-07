import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv

#paths
DATA_DIR = "."
MODEL_DIR = "models"
OUTPUT_CSV = "results/evaluation_all_drift.csv"
LATENCY_RUNS = 100

#model list
KERAS_MODELS = [
    {"name": "CNN-LSTM Base", "file": "cnn_lstm1.keras"},
    {"name": "CNN Base", "file": "baseline_cnn.h5"},
    {"name": "CNN-LSTM Pruned", "file": "cnn_lstm_pruned70.keras"},
    {"name": "CNN Pruned", "file": "baseline_cnn_pruned70.h5"},
]

TFLITE_MODELS = [
    {"name": "CNN-LSTM FP32", "file": "tflite_lstm/lstm_fp32.tflite"},
    {"name": "CNN-LSTM Dynamic", "file": "tflite_lstm/lstm_dynamic.tflite"},
    {"name": "CNN-LSTM FP16", "file": "tflite_lstm/lstm_fp16.tflite"},
    {"name": "CNN-LSTM Int8", "file": "cnn_lstm1_quant.tflite"},
    {"name": "CNN FP32", "file": "tflite_cnn/cnn_fp32.tflite"},
    {"name": "CNN Dynamic", "file": "tflite_cnn/cnn_dynamic.tflite"},
    {"name": "CNN FP16", "file": "tflite_cnn/cnn_fp16.tflite"},
    {"name": "CNN Int8", "file": "cnn_8bit.tflite"},
    {"name": "Pruned CNN-LSTM FP32", "file": "tflitelstm_prune70/prune70lstm_fp32.tflite"},
    {"name": "Pruned CNN-LSTM Dynamic", "file": "tflitelstm_prune70/prune70lstm_dynamic.tflite"},
    {"name": "Pruned CNN-LSTM FP16", "file": "tflitelstm_prune70/prune70lstm_fp16.tflite"},
    {"name": "Pruned CNN FP32", "file": "tflitecnn_prune70/prune70cnn_fp32.tflite"},
    {"name": "Pruned CNN Dynamic", "file": "tflitecnn_prune70/prune70cnn_dynamic.tflite"},
    {"name": "Pruned CNN FP16", "file": "tflitecnn_prune70/prune70cnn_fp16.tflite"},
]

#load dat
def load_test_data():
    X = np.load("X_drift1.npy")
    y = np.load("y_drift1.npy")
    return X, y

#metrics
def compute_metrics(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    y_true = y_true.flatten().astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
    }

#evaluation for keras
def evaluate_keras(model_info, X, y):
    path = os.path.join(MODEL_DIR, model_info["file"])
    model = tf.keras.models.load_model(path)

    y_pred = model.predict(X, verbose=0)
    metrics = compute_metrics(y, y_pred)

    sample = X[:1]
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        _ = model(sample, training=False) #changed from model.predict so it would do it one by one like ftlite
        times.append((time.perf_counter() - start) * 1000)

    return {
        "name": model_info["name"],
        "type": "Keras",
        "size_kb": os.path.getsize(path) / 1024,
        "latency_ms": np.mean(times[10:]), #mean to drop the warmups and hopefully less bias, added to tflite too
        **metrics
    }

#evaluation for ftlite
def evaluate_tflite(model_info, X, y):
    path = os.path.join(MODEL_DIR, model_info["file"])
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    dtype = inp[0]["dtype"] #mathe the quant format 

    preds = []
    for sample in X:
        s = sample.reshape(1, -1).astype(dtype)
        interpreter.set_tensor(inp[0]["index"], s)
        interpreter.invoke() 
        preds.append(interpreter.get_tensor(out[0]["index"])[0][0])

    metrics = compute_metrics(y, np.array(preds))

    single = X[:1].astype(dtype)
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(inp[0]["index"], single)
        interpreter.invoke()
        interpreter.get_tensor(out[0]["index"])
        times.append((time.perf_counter() - start) * 1000)

    return {
        "name": model_info["name"],
        "type": "TFLite",
        "size_kb": os.path.getsize(path) / 1024,
        "latency_ms": np.mean(times[10:]),
        **metrics
    }

def main():
    os.makedirs("results", exist_ok=True)
    X, y = load_test_data()
    results = []

    for m in KERAS_MODELS:
        results.append(evaluate_keras(m, X, y))

    for m in TFLITE_MODELS:
        results.append(evaluate_tflite(m, X, y))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
