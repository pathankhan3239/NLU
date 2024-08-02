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
            # Convert input to a list
            input_list = [user_input]

            # Get sentiment analysis result
            prediction = classifier(input_list, candidate_labels=labels)

            # Extract results from the prediction list
            pred = prediction[0]
            pred_labels = pred['labels']
            pred_scores = pred['scores']

            # Extract label and score from prediction
            prediction_label = pred_labels[0]
            prediction_score = pred_scores[0]

            st.write(f"**Label:** {prediction_label}, **Score:** {prediction_score:.4f}")

            # Show detailed probabilities for all labels
            scores_df = pd.DataFrame({
                "Label": pred_labels,
                "Score": pred_scores
            })
            st.write("**Detailed Scores:**")
            st.write(scores_df)

            # Store the input and prediction in history
            st.session_state['history'].append({
                "Input": user_input,
                "Prediction": prediction_label,
                "Scores": pred_scores
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
            def shap_predict(texts):
                return predict_proba(texts)

            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(shap_predict, masker)
            shap_values = explainer(input_list)

            st.write("Explanation:")
            shap.initjs()
            fig = shap.plots.text(shap_values[0])
            st.pyplot(fig)

        except Exception as e:
            st.write(f"Error: {e}")
