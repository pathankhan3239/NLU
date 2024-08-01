import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import shap
import torch

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Get sentiment analysis result
    result = sentiment_analysis(user_input)[0]
    st.write(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")

    # Explain the result using SHAP
    def predict(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    
    explainer = shap.Explainer(predict, tokenizer)
    shap_values = explainer([user_input])

    st.write("Explanation:")
    st_shap = shap.plots.text(shap_values[0], display=False)
    st.pyplot(st_shap)
