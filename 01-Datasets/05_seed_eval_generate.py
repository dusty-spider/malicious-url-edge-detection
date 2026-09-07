import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import sys
from pathlib import Path
#tokenisation and canonicalisation functions from Common
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.preprocessing import tokenize, canonicalize_url
# same tokenisation used for reproductability from Common

SEED = 8  # change to random one if needed (FIX NO HARD CODE)
MAX_LEN = 200

# split part taken from datasplit.py
dataset_main = pd.read_csv("final_dataset1.csv")
X = dataset_main["url"]
y = dataset_main["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

test_df = pd.DataFrame({"url": X_test, "label": y_test})

test_df["url"] = test_df["url"].astype(str).apply(canonicalize_url)

# reproduction
with open("char_to_idx_reproduce.json", "r") as f:
    char_to_idx = json.load(f)

X_test_tok = pad_sequences(
    test_df["url"].apply(tokenize),
    maxlen=MAX_LEN, padding="post", truncating="post"
)
y_test_arr = test_df["label"].values

# save up here
np.save(f"X_test_seed8.npy", X_test_tok)
np.save(f"y_test_seed8.npy", y_test_arr)

print(f"Seed test ready!!!")
