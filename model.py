# app/model.py
import joblib
from pathlib import Path

MODEL_PATH = Path("models/sentiment_baseline_tfidf_lr.joblib")

model = joblib.load(MODEL_PATH)

def predict_sentiment(text: str):
    text = text.strip()[:500]  # safety limit

    if not text:
        return {
            "sentiment": "Neutral",
            "confidence": 0.0
        }

    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]

    return {
        "sentiment": pred,
        "confidence": round(float(max(proba)), 4)
    }
