import pandas as pd

data_kaggle = pd.read_csv("kaggle_clean.csv")
data_tranco = pd.read_csv("tranco_clean.csv")
data_phish = pd.read_csv("phishtank_clean.csv")

# merge all
all_dat = pd.concat(
    [data_kaggle, data_tranco, data_phish],
    ignore_index=True
)

#dedublication
print("Lenght before the deduplication:", len(all_dat))
all_dat = all_dat.drop_duplicates(subset="url")
print("Lenght after deduplication:", len(all_dat))

print("\nLabel distribution:")
print(all_dat["label"].value_counts())
print("\nSource distribution:")
print(all_dat["source"].value_counts())
print("\nSample rows:")
print(all_dat.sample(5))

all_dat.to_csv(
    "final_dataset1.csv",
    index=False
)

print("Dataset saved :)")

ben = (all_dat["label"] == 0).sum()
mal = (all_dat["label"] == 1).sum()

print(f"Benign: {ben}")
print(f"Malicious: {mal}")
print(f"Ratio (benign:malicious): {ben/mal:.2f}") #for future balancing

