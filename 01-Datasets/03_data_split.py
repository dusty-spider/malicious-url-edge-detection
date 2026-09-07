import pandas as pd
from sklearn.model_selection import train_test_split

dataset_main = pd.read_csv("final_dataset1.csv")

# label the rows for dataframes division
X = dataset_main["url"]
y = dataset_main["label"]

# first split: 70% train, 30% as temporary for now as the function can't do 70 15 15 at once
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=69 #seed, later used to generate another test material for evaluation
)

# second split (dividing the 30 from first) to 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=69
)

# reconstructing the dataframes for the model training for x and y 
train_dataframe = pd.DataFrame({"url": X_train, "label": y_train})
val_dataframe   = pd.DataFrame({"url": X_val,   "label": y_val})
test_dataframe  = pd.DataFrame({"url": X_test,  "label": y_test})

train_dataframe.to_csv("1training30.csv", index=False)
val_dataframe.to_csv("1validation15.csv", index=False)
test_dataframe.to_csv("1testing15.csv", index=False)

# recheck if it went right, if not the size will not line up
def check_split(name, dataframe):
    print(f"\n{name}")
    print("Size:", len(dataframe))
    print(dataframe["label"].value_counts(normalize=True))

check_split("Training", train_dataframe)
check_split("Validation", val_dataframe)
check_split("Testing", test_dataframe)
