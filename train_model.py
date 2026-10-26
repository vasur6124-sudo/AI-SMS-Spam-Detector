import pandas as pd
from sklearn.model_selection import train_test_split

#Load our dataset
data=pd.read_csv(
    "dataset\SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label","message"]
)

#Separate messages and their labels

X=data["message"]
y=data["label"]

print("Messages:")
print(X.head())

print("\nLabels:")
print(y.head())

#Split the dataset into training and testing data 
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
test_size=0.2,
random_state=42,
stratify=y
)

print("Total Messages:",len(X))
print("Training Messages:",len(X_train))
print("Testing Messages:",len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

#create a TF-IDF converter
tfidf =  TfidfVectorizer()

#Learn vocabulary and convert training messages into numbers
X_train_tfidf = tfidf.fit_transform(X_train)

#Convert testing messages using the same learned vocabulary 
X_test_tfidf = tfidf.transform(X_test)

print("Training data shape :",X_train_tfidf.shape)
print("Testing data shape :",X_test_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB

#Create the machine learning model
model = MultinomialNB()

#Train the model using SMS messages and their correct labels
model.fit(X_train_tfidf,y_train)
  #X_train comtains the numerical features of the training messages and y_train contains the correct labels:ham or spam
print("Model trainig completed successfully")

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#predict labels for the testing messages
y_pred = model.predict(X_test_tfidf)

#calculate accuracy
accuracy = accuracy_score(y_test,y_pred)
print("\nModel Accuracy:",round(accuracy*100,2),"%")

#Display detailed performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test the model with ya new SMS message
message = input("\nEnter an SMS message: ")

# Convert the messsage into Tf-Idf feature
message_tfidf = tfidf.transform([message])

#Predict the category
prediction = model.predict(message_tfidf)[0]

if prediction == "spam":
    print("Prediction:Spam")
else:
    print("Prediction:Ham(Genuine Message)")

import pickle

#saving the trained model
with open("model.pkl","wb")as file:
    pickle.dump(model,file)

# Save the TF-IDF converter

with open("vectorizer.pkl","wb")as file:
    pickle.dump(tfidf,file)

print("Model and vectorizer saved successfully!")

