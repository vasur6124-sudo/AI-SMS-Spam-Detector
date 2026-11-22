
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Separate messages and labels
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# Improved TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

# Convert text into numerical features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)


# Create two machine learning models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
}

# Train and evaluate both models
for name, model in models.items():

    print("\n==============================")
    print("Model:", name)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", round(
        accuracy_score(y_test, y_pred) * 100, 2
    ), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

import pickle

# Train the final Logistic Regression model
best_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

best_model.fit(X_train_tfidf, y_train)

# Save model and vectorizer together
with open("model_v2.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("vectorizer_v2.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("\nVersion 2 model saved successfully!")
