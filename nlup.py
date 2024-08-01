import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap
import torch
import numpy as np

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to make predictions
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

# SHAP explainer initialization
explainer = shap.Explainer(predict_proba, masker=shap.maskers.Text(tokenizer))

# Set up Streamlit app
st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Get sentiment analysis result
    prediction = classifier(user_input)[0]
    st.write(f"Sentiment: {prediction['label']}, Score: {prediction['score']:.4f}")

    # Explain the result using SHAP
    shap_values = explainer([user_input])

    st.write("Explanation:")
    st_shap = shap.plots.text(shap_values[0])
    st.pyplot(st_shap)
