import pandas as pd

# Load the SMS dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first five messages
print(data.head())

# Count spam and legitimate messages
print("\nMessage counts:")
print(data["label"].value_counts())

# Display the dataset size
print("\nTotal messages:", len(data))