import json
import re
import numpy as np
from ai_edge_litert.interpreter import Interpreter  # Runtime used on the Pi


# Configuration
MODEL_PATH    = "lstm_dynamic.tflite"          # swap to whichever deployed model you copied onto the Pi
VOCAB_PATH    = "char_to_idx_reproduce.json"
MAX_LEN       = 200
THRESHOLD     = 0.5


# Load vocab once at import time
with open(VOCAB_PATH, "r") as f:
    char_to_idx = json.load(f)

# Load TFLite interpreter once at import time
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def canonicalize_url(url: str) -> str:
    # Matches the canonicalisation used during training.
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    return url


def tokenize(url: str) -> list:
    # Matches the tokenisation used during training.
    return [char_to_idx.get(char, 1) for char in url]   # 1 = <UNK>


def preprocess(url: str) -> np.ndarray:
    url     = canonicalize_url(url)
    tokens  = tokenize(url)
    # Manual pad / truncate to match keras pad_sequences(padding='post')
    tokens  = tokens[:MAX_LEN]
    tokens  += [0] * (MAX_LEN - len(tokens))
    return np.array([tokens], dtype=np.float32)


def predict_url(url: str) -> dict:
    import time
    x = preprocess(url)
    interpreter.set_tensor(input_details[0]['index'], x)
    t0 = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    return {
        "url":        url,
        "verdict":    "malicious" if score >= THRESHOLD else "benign",
        "confidence": round(score, 4),
        "latency_ms": round(latency_ms, 3),
    }
