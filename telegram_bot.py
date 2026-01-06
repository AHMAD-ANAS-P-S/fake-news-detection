from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import joblib
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# ===== LOAD MODEL =====
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===== TOKENS =====
TOKEN = os.getenv("TELEGRAM_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ===== START COMMAND =====
def start(update, context):
    update.message.reply_text(
        "Welcome to Fake News Detection Bot 🤖\n"
        "Send any news text.\n\n"
        "Commands:\n"
        "/scrape - latest BBC news\n"
        "/newsbydate YYYY-MM-DD"
    )

# ===== PREDICTION FUNCTION =====
def predict(text):
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    label = "Real" if pred == 1 else "Fake"
    confidence = round(max(prob) * 100, 2)
    return label, confidence

# ===== HANDLE NORMAL TEXT =====
def handle_message(update, context):
    label, confidence = predict(update.message.text)
    update.message.reply_text(
        f"Prediction: {label}\nConfidence: {confidence}%"
    )

# ===== /SCRAPE COMMAND =====
def scrape(update, context):
    url = "https://www.bbc.com/news"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")


    # BBC uses these classes for real headlines
    headlines = soup.select(
        "h2[data-testid='card-headline'], "
        "h3[data-testid='card-headline']"
    )

    sent = 0
    for h in headlines:
        text = h.get_text(strip=True)

    # skip useless BBC banner text
        if "BBC is in multiple languages" in text:
            continue

        if len(text) < 40:
            continue


        label, _ = predict(text)
        update.message.reply_text(
            f"Headline: {text}\nPrediction: {label}"
        )

        sent += 1
        if sent == 5:
            break

    if sent == 0:
        update.message.reply_text("No headlines found. BBC page layout changed.")


# ===== /NEWSBYDATE COMMAND =====
def news_by_date(update, context):
    if len(context.args) != 1:
        update.message.reply_text(
            "Use like this:\n/newsbydate YYYY-MM-DD"
        )
        return

    date = context.args[0]

    # ---- DATE LIMIT CHECK ----
    allowed_from = datetime(2025, 12, 5).date()
    requested_date = datetime.strptime(date, "%Y-%m-%d").date()

    if requested_date < allowed_from:
        update.message.reply_text(
            "This date is too old for the free NewsAPI plan.\n"
            "Try a recent date (from 2025-12-05 onwards)."
        )
        return

    # ---- MAIN QUERY ----
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=news&"
        f"from={date}&to={date}&"
        f"language=en&sortBy=popularity&"
        f"apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url).json()
    articles = response.get("articles", [])

    # ---- FALLBACK ----
    if not articles:
        update.message.reply_text(
            "No news found for this date.\nShowing latest news instead."
        )

        fallback_url = (
            f"https://newsapi.org/v2/top-headlines?"
            f"language=en&apiKey={NEWS_API_KEY}"
        )

        fallback = requests.get(fallback_url).json()
        articles = fallback.get("articles", [])

    # ---- DISPLAY ----
    for article in articles[:5]:
        title = article["title"]
        label, _ = predict(title)
        update.message.reply_text(
            f"Title: {title}\nPrediction: {label}"
        )

# ===== BOT SETUP =====
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("scrape", scrape))
dp.add_handler(CommandHandler("newsbydate", news_by_date))
dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# ===== START BOT =====
updater.start_polling()
updater.idle()
