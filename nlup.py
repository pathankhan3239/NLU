import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import shap

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to make predictions
def predict(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

# Set up Streamlit app
st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Get sentiment analysis result
    prediction = predict([user_input])[0]
    sentiment = "POSITIVE" if prediction[1] > prediction[0] else "NEGATIVE"
    score = max(prediction)
    st.write(f"Sentiment: {sentiment}, Score: {score:.4f}")

    # Explain the result using SHAP
    explainer = shap.Explainer(predict, tokenizer)
    shap_values = explainer([user_input], check_additivity=False)

    st.write("Explanation:")
    shap_text = shap.plots.text(shap_values[0])
    st.pyplot(shap_text)
