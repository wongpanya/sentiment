# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_sentiment

app = FastAPI(
    title="Thai Sentiment API",
    description="Sentiment analysis for Thai social media text (Lab)",
    version="1.0"
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: TextInput):
    return predict_sentiment(input.text)
