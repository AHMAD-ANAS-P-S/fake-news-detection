from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        news_text = request.form["news_text"]

        vec = vectorizer.transform([news_text])
        prob = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]

        prediction = "Real" if pred == 1 else "Fake"
        confidence = round(max(prob) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
