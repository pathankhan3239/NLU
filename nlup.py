import streamlit as st
from transformers import pipeline

# Load pre-trained sentiment analysis pipeline with RoBERTa
sentiment_analysis = pipeline("sentiment-analysis", model="roberta-large-mnli")

st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    result = sentiment_analysis(user_input)[0]
    st.write(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
    
    # Explain the result (placeholder for actual explanation code)
    st.write("Explanation:", explanation)
