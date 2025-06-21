from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

class TextIn(BaseModel):
    text: str

app = FastAPI()
model = joblib.load("models/svm_goemotions.pkl")

@app.post("/predict")
def predict(data: TextIn):
    pred = model.predict([data.text])[0]
    prob = model.predict_proba([data.text])[0]
    return {"emotion": pred, "confidence": float(max(prob))}