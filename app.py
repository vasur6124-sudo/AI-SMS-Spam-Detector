
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load Version 2 model
with open("model_v2.pkl", "rb") as file:
    model = pickle.load(file)

# Load Version 2 vectorizer
with open("vectorizer_v2.pkl", "rb") as file:
    tfidf = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            message_tfidf = tfidf.transform([message])
            prediction = model.predict(message_tfidf)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=False)
