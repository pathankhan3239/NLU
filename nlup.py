import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define labels for sentiment analysis
labels = ["positive", "neutral", "negative"]

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
            prediction = classifier(user_input, candidate_labels=labels)
            st.write(f"Label: {prediction['labels'][0]}, Score: {prediction['scores'][0]:.4f}")
        except Exception as e:
            st.write(f"Error: {e}")
