import pandas as pd
import re
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import sys
from pathlib import Path
#tokenisation and canonicalisation functions from Common
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Common.preprocessing import canonicalize_url, tokenize


# load splits
train_df = pd.read_csv("1training30.csv")
val_df   = pd.read_csv("1validation15.csv")
test_df  = pd.read_csv("1testing15.csv")

#apply the strip 
for df in [train_df, val_df, test_df]:
    df["url"] = df["url"].astype(str).apply(canonicalize_url)


# build character vocabulary for training data only 
all_chars = set("".join(train_df["url"].values))
# 0 = padding
# 1 = unknown
char_to_idx = {char: idx + 2 for idx, char in enumerate(sorted(all_chars))}
char_to_idx["<PAD>"] = 0
char_to_idx["<UNK>"] = 1

vocab_size = len(char_to_idx)
print(f"Vocabulary size: {vocab_size}")

# tokenisation application
X_train = train_df["url"].apply(tokenize)
X_val   = val_df["url"].apply(tokenize)
X_test  = test_df["url"].apply(tokenize)

# padding max lenght as urlnet base for uses 200 for default
MAX_LEN = 200

X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="post", truncating="post")
X_val   = pad_sequences(X_val,   maxlen=MAX_LEN, padding="post", truncating="post")
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding="post", truncating="post")

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

# save outputs
np.save("X_train1.npy", X_train)
np.save("X_val1.npy", X_val)
np.save("X_test1.npy", X_test)

np.save("y_train1.npy", y_train)
np.save("y_val1.npy", y_val)
np.save("y_test1.npy", y_test)

# save vocab for reproduction and for pi at the deployment stage, and the drift too
import json
with open("char_to_idx_reproduce.json", "w") as f:
    json.dump(char_to_idx, f)

print("Tokenisation and padding complete!!!! :DDD")
