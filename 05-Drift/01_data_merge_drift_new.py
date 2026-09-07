import pandas as pd

# load original dataset
df_original = pd.read_csv("final_dataset1.csv")
original_urls = set(df_original["url"].str.strip().str.lower())
                    
# load cleaned datasets
df_all = pd.read_csv("phiusiil_clean_drift.csv")


print("Lenght before deduplication:", len(df_all))

# drop duplicate URLs
df_all = df_all.drop_duplicates(subset="url")
print("Lenght after deduplication:", len(df_all))

# remove URLs that appeared in original training data
df_all["url_lower"] = df_all["url"].str.strip().str.lower()
df_drift = df_all[~df_all["url_lower"].isin(original_urls)].drop(columns="url_lower")
print("Length after removing original dataset URLs:", len(df_drift))

print("\nLabel distribution:")
print(df_drift["label"].value_counts())

print("\nSource distribution:")
print(df_all["source"].value_counts())

print("\nSample rows:")
print(df_all.sample(5))

df_drift.to_csv(
    "drift_dataset_new.csv",
    index=False
)
print("Dataset saved :)")

benign = (df_all["label"] == 0).sum()
malicious = (df_all["label"] == 1).sum()

print(f"Benign: {benign}")
print(f"Malicious: {malicious}")
print(f"Ratio (benign:malicious): {benign/malicious:.2f}")
