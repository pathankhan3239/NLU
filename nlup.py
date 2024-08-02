import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to make predictions
def predict_sentiment(text):
    # Tokenize and prepare tensors
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs

# Set up Streamlit app
st.title("Efficient Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Check if input is not empty
    if user_input.strip() == "":
        st.write("Please enter a review.")
    else:
        try:
            # Get sentiment analysis result
            prediction = classifier(user_input)[0]
            st.write(f"Sentiment: {prediction['label']}, Score: {prediction['score']:.4f}")
        except Exception as e:
            st.write(f"Error: {e}")
