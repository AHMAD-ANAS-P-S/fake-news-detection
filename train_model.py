import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Loading dataset...")
data = pd.read_csv("fake_news_dataset.csv")

# VERY IMPORTANT: check column names
# If label is 0/1 → keep as is
# If label is FAKE/REAL → convert
if data['label'].dtype == object:
    data['label'] = data['label'].map({'FAKE': 0, 'REAL': 1})

X = data['text']
y = data['label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("MODEL TRAINED AND SAVED SUCCESSFULLY")
