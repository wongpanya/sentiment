from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib
from pathlib import Path

app = FastAPI(
    title="Thai Sentiment Demo",
    description="Baseline TF-IDF + Logistic Regression (Lab/Demo)",
    version="1.0"
)

# ---- Load model (same folder as main.py) ----
MODEL_PATH = Path("sentiment_baseline_tfidf_lr.joblib")
model = joblib.load(MODEL_PATH)

class TextInput(BaseModel):
    text: str

def predict_sentiment(text: str):
    text = (text or "").strip()[:500]
    if not text:
        return {"sentiment": "Neutral", "confidence": 0.0}

    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    return {"sentiment": pred, "confidence": round(float(max(proba)), 4)}

# ---- Health ----
@app.get("/")
def health():
    return {"status": "ok", "demo": "/demo", "docs": "/docs"}

# ---- API ----
@app.post("/predict")
def predict(payload: TextInput):
    return predict_sentiment(payload.text)

@app.get("/predict")
def predict_info():
    return {
        "message": "Use POST /predict with JSON body",
        "example": {"text": "เลือกตั้งกี่รอบก็เหมือนเดิม เบียว 🙄"}
    }

# ---- Demo page (embedded HTML) ----
DEMO_HTML = """
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Thai Sentiment Demo</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f6f7fb; margin:0; padding:0; }
    .wrap { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; }
    .card { background:#fff; width:520px; max-width:95vw; padding:22px; border-radius:14px;
            box-shadow:0 10px 25px rgba(0,0,0,.10); }
    h2 { margin:0 0 12px; text-align:center; font-weight:700; }
    textarea { width:100%; height:120px; padding:10px; font-size:14px; border-radius:10px;
               border:1px solid #d1d5db; resize:none; box-sizing:border-box; }
    button { margin-top:12px; width:100%; padding:11px; background:#2563eb; color:#fff; border:none;
             border-radius:10px; font-size:15px; cursor:pointer; }
    button:hover { background:#1e40af; }
    .meta { margin-top:10px; font-size:12px; color:#6b7280; text-align:center; }
    .result { margin-top:16px; text-align:center; font-size:16px; }
    .negative { color:#dc2626; }
    .neutral { color:#6b7280; }
    .positive { color:#16a34a; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#f3f4f6; margin-top:6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2>Thai Sentiment Analysis</h2>
      <textarea id="textInput" placeholder="พิมพ์ความคิดเห็นที่นี่..."></textarea>
      <button onclick="analyze()">Analyze</button>
      <div class="meta">API: <code>POST /predict</code> • Docs: <code>/docs</code></div>
      <div class="result" id="result"></div>
    </div>
  </div>

<script>
async function analyze() {
  const el = document.getElementById("textInput");
  const text = (el.value || "").trim();
  if (!text) { alert("กรุณาพิมพ์ข้อความ"); return; }

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });

  const out = await res.json();
  const sentiment = out.sentiment || "Neutral";
  const conf = out.confidence ?? 0;

  let cls = "neutral";
  if (sentiment === "Positive") cls = "positive";
  if (sentiment === "Negative") cls = "negative";

  document.getElementById("result").innerHTML = `
    <div class="${cls}">
      <div style="font-size:22px; font-weight:700;">${sentiment}</div>
      <div class="pill">Confidence: ${(conf*100).toFixed(1)}%</div>
    </div>
  `;
}
</script>
</body>
</html>
"""

@app.get("/demo", response_class=HTMLResponse)
def demo():
    return DEMO_HTML
