from fastapi import FastAPI
from pydantic import BaseModel
from backend import predict

app = FastAPI()

class Input(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict/")
async def predict_sentiment(input: Input):
    prediction = predict(input.text)
    return {"text": input.text, "sentiment": prediction}
