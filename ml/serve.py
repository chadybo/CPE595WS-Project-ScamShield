"""
ScamShield model service. The Express app calls this to classify a message.

Run (from the ml/ folder, after python train_baseline.py):
    uvicorn serve:app --port 8000
"""
import re
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "models" / "baseline_app.joblib"
if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run `python train_baseline.py` in the ml/ folder first.")

model = joblib.load(MODEL_PATH)
vectorizer = model.named_steps["features"].named_transformers_["text"]
coefs = model.named_steps["clf"].coef_[0]
vocab = vectorizer.get_feature_names_out()

URGENCY = re.compile(r"\b(urgent|immediately|act now|limited|expires?|final notice|asap)\b", re.I)
MONEY = re.compile(r"\b(cash|prize|win|winner|free|money|\$\d+|reward|refund|gift card)\b", re.I)
LINK = re.compile(r"https?://|www\.", re.I)
CREDENTIALS = re.compile(r"\b(password|verify your account|login|ssn|bank account)\b", re.I)

app = FastAPI(title="ScamShield model service")


class Message(BaseModel):
    subject: str = ""
    content: str = Field(min_length=1, max_length=20000)


def warning_signs(subject: str, content: str, top_terms: list[str]) -> list[str]:
    text = f"{subject} {content}"
    signs = []
    if URGENCY.search(text):
        signs.append("Uses urgent or time-pressure language")
    if MONEY.search(text):
        signs.append("Mentions money, prizes, or free offers")
    if LINK.search(text):
        signs.append("Contains a link. Check where it really goes before clicking")
    if CREDENTIALS.search(text):
        signs.append("Asks about passwords, logins, or account details")
    if top_terms:
        signs.append("Words the model flagged: " + ", ".join(top_terms))
    return signs


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(msg: Message):
    row = pd.DataFrame({"text": [f"{msg.subject} {msg.content}"]})
    prob = float(model.predict_proba(row)[0, 1])
    label = "phishing" if prob >= 0.5 else "legit"

    # Terms in this message that pushed the score toward phishing
    vec = vectorizer.transform(row["text"])
    contrib = vec.multiply(coefs).tocsr()
    pairs = sorted(zip(contrib.indices, contrib.data), key=lambda p: p[1], reverse=True)
    top_terms = [vocab[i] for i, v in pairs if v > 0][:5]

    return {
        "label": label,
        "probability": round(prob, 4),
        "warning_signs": warning_signs(msg.subject, msg.content, top_terms) if label == "phishing" else [],
    }
