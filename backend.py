import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load model and tokenizer only once at startup
config = PeftConfig.from_pretrained("rabindra-sss/sentiment-distilbert")
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = PeftModel.from_pretrained(base_model, "rabindra-sss/sentiment-distilbert", config=config)
tokenizer = AutoTokenizer.from_pretrained("rabindra-sss/sentiment-distilbert")

# Ensure model is in evaluation mode for inference
model.eval()

# Define id2label mappings
id2label = {0: "Negative", 1: "Positive"}

def predict(text: str) -> str:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Run the model to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to predicted class
    predictions = torch.argmax(logits, dim=-1)
    predicted_label = id2label[predictions.item()]

    return predicted_label
