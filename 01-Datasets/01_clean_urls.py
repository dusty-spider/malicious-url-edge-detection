import pandas as pd


# function to clean the urls up
def clean_url(url):
    if pd.isna(url):
        return None
    return url.strip().lower().rstrip("/")


# Kaggle dataset
df = pd.read_csv("malicious_phish.csv")
df = df[["url", "type"]]
df["label"] = df["type"].apply(lambda x: 0 if x == "benign" else 1) # erase kaggles other labels like malware or phishware
df["source"] = "kaggle"
df["url"] = df["url"].apply(clean_url) # function application 
df = df.dropna(subset=["url"]) #need to do to all of them just in case
df = df[["url", "label", "source"]]
df.to_csv("kaggle_clean.csv", index=False)

print(df.head())
print(df["label"].value_counts())


# tranco preprocess
df = pd.read_csv("tranco_YL7JG.csv", header=None, names=["rank", "domain"])
df = df.rename(columns={"domain": "url"}) 
df["label"] = 0 #bening all 0
df["source"] = "tranco"
df["url"] = df["url"].apply(clean_url)
df = df.dropna(subset=["url"]) 
df = df[["url", "label", "source"]]
df.to_csv("tranco_clean.csv", index=False) #save up

print(df.head())
print(df["label"].value_counts())


# phishtank preprocess
df = pd.read_csv("verified_online.csv")
df = df[df["verified"] == "yes"] #only verified as some are not confirmed 
df = df[["url"]]
df["label"] = 1
df["source"] = "phishtank"
df["url"] = df["url"].apply(clean_url)
df = df.dropna(subset=["url"])
df = df[["url", "label", "source"]]
df.to_csv("phishtank_clean.csv", index=False)

print(df.head())
print(df["label"].value_counts())
