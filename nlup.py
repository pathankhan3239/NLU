import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap
import torch
import numpy as np

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to make predictions
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

# SHAP requires a function that accepts raw text and returns predictions
def shap_predict(texts):
    predictions = predict_proba(texts)
    return predictions

# Define a masker function
def masker(input_strings):
    encoded_inputs = tokenizer(input_strings, return_tensors='pt', padding=True, truncation=True)
    attention_mask = encoded_inputs['attention_mask'].numpy()
    return attention_mask

# Set up Streamlit app
st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Get sentiment analysis result
    prediction = shap_predict([user_input])[0]
    sentiment = "POSITIVE" if prediction[2] > max(prediction[0], prediction[1]) else "NEGATIVE"
    score = max(prediction)
    st.write(f"Sentiment: {sentiment}, Score: {score:.4f}")

    # Explain the result using SHAP
    explainer = shap.Explainer(shap_predict, masker)
    shap_values = explainer([user_input])

    st.write("Explanation:")
    fig = shap.plots.text(shap_values[0])
    st.pyplot(fig)
