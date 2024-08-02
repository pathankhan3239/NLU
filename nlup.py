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

# SHAP explainer initialization
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(lambda x: classifier(x)[0]['score'], masker)

# Set up Streamlit app
st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Check if input is not empty
    if user_input.strip() == "":
        st.write("Please enter a review.")
    else:
        try:
            # Get sentiment analysis result
            prediction = classifier(user_input)[0]
            sentiment = prediction['label']
            score = prediction['score']

            # Explain the result using SHAP
            shap_values = explainer(user_input)  # Pass user_input directly

            # Display results
            st.write(f"Sentiment: {sentiment}, Score: {score:.4f}")
            st.write("Explanation:")
            fig = shap.plots.text(shap_values)
            st.pyplot(fig)

        except Exception as e:
            st.write(f"Error: {e}")
