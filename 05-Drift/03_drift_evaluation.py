import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv

#paths
DATA_DIR = "."
MODEL_DIR = "models"
OUTPUT_CSV = "drift_result_new.csv"
LATENCY_RUNS = 100

TFLITE_MODELS = [
    {"name": "CNN-LSTM Dynamic", "file": "tflite_lstm/lstm_dynamic.tflite"},
    {"name": "CNN Dynamic", "file": "tflite_cnn/cnn_dynamic.tflite"},
    {"name": "Pruned CNN-LSTM Dynamic", "file": "tflitelstm_prune70/prune70lstm_dynamic.tflite"},
    {"name": "Pruned CNN Dynamic", "file": "tflitecnn_prune70/prune70cnn_dynamic.tflite"},
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
        interpreter.invoke() #diff from predict, might use it on pi later
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

    for m in TFLITE_MODELS:
        results.append(evaluate_tflite(m, X, y))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
