import pandas as pd
import numpy as np
import json
import re
from keras.preprocessing.sequence import pad_sequences

import sys
from pathlib import Path
# og tokenisation and canonicalisation
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Common.preprocessing import canonicalize_url, tokenize

# the og vocabulary
with open("char_to_idx_reproduce.json", "r") as f:
    char_to_idx = json.load(f)


df = pd.read_csv("drift_dataset_new.csv")
df["url"] = df["url"].astype(str).apply(canonicalize_url)

X_drift = pad_sequences(df["url"].apply(tokenize), maxlen=200, padding="post", truncating="post")
y_drift = df["label"].values

np.save("X_drift1.npy", X_drift)
np.save("y_drift1.npy", y_drift)

print(f"Drift dataset ready to use: {X_drift.shape}")