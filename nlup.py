import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap
import torch
import pandas as pd
import altair as alt

# Load pre-trained sentiment analysis model and tokenizer
model_name = "roberta-large-mnli"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define labels for sentiment analysis
labels = ["positive", "neutral", "negative"]

# Function to make predictions
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

# Set up Streamlit app
st.title("Advanced Sentiment Analysis Tool")
user_input = st.text_area("Enter a review:")

# Store the history of inputs and their predictions
if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button("Analyze"):
    # Check if input is not empty
    if user_input.strip() == "":
        st.write("Please enter a review.")
    else:
        try:
            # Get sentiment analysis result
            prediction = classifier(user_input, candidate_labels=labels)
            st.write(f"**Label:** {prediction['labels'][0]}, **Score:** {prediction['scores'][0]:.4f}")

            # Show detailed probabilities for all labels
            st.write("**Detailed Scores:**")
            scores_df = pd.DataFrame({
                "Label": prediction['labels'],
                "Score": prediction['scores']
            })
            st.write(scores_df)

            # Store the input and prediction in history
            st.session_state['history'].append({
                "Input": user_input,
                "Prediction": prediction['labels'][0],
                "Scores": prediction['scores']
            })

            # Display the history
            st.write("### History")
            history_df = pd.DataFrame(st.session_state['history'])
            st.write(history_df)

            # Visualization: Bar chart of the sentiment scores
            st.write("### Sentiment Score Visualization")
            chart = alt.Chart(scores_df).mark_bar().encode(
                x='Label',
                y='Score',
                color='Label'
            ).properties(
                width=alt.Step(80)  # controls the width of the bars
            )
            st.altair_chart(chart)

            # Explain the result using SHAP
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(predict_proba, masker)
            shap_values = explainer([user_input])

            st.write("Explanation:")
            fig = shap.plots.text(shap_values[0])
            st.pyplot(fig)

        except Exception as e:
            st.write(f"Error: {e}")
