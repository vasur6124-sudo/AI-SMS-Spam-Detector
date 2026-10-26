# AI SMS Spam Detector

## Project Overview
This project uses Machine Learning and Natural Language Processing (NLP) to classify SMS messages as Spam or Ham (Not Spam).

It uses TF-IDF to convert text messages into numerical features and a Multinomial Naive Bayes classifier to make predictions.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Flask
- HTML and CSS

## Features
- Classifies SMS messages as Spam or Ham.
- Uses TF-IDF for text feature extraction.
- Trains a Naive Bayes machine-learning model.
- Provides a web interface using Flask.

## Model Performance
The model achieved approximately 96.05% accuracy on the test split used during development. It did not identify every spam message, so predictions are not guaranteed to be correct.

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Place the SMS Spam Collection dataset file named SMSSpamCollection inside the dataset folder.

3. Train the model:
   python train_model.py

4. Start the web application:
   python app.py

5. Open http://127.0.0.1:5000 in your browser.

## Learning Purpose
This project helped me practise Python, text preprocessing, machine learning classification, model evaluation, and Flask web development.
