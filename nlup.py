import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap
import torch
import pandas as pd
import altair as alt

# -------------------- Model and Tokenizer Setup --------------------
model_name = "roberta-large-mnli"  # Zero-shot classification model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Define sentiment labels
labels = ["positive", "neutral", "negative"]

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Sentiment Analysis Tool", layout="centered")
st.title("Advanced Sentiment Analysis Tool")
st.write("Enter a sentence or review below to analyze its sentiment:")

user_input = st.text_area("Enter a review:")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# -------------------- Prediction and Display --------------------
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        try:
            # Run zero-shot classification
            prediction = classifier(user_input, candidate_labels=labels)
            pred_labels = prediction["labels"]
            pred_scores = prediction["scores"]
            prediction_label = pred_labels[0]
            prediction_score = pred_scores[0]

            # Display prediction
            st.success(f"**Predicted Sentiment:** {prediction_label} (Score: {prediction_score:.4f})")

            # Display score breakdown
            scores_df = pd.DataFrame({
                "Label": pred_labels,
                "Score": pred_scores
            })
            st.subheader("Detailed Scores")
            st.dataframe(scores_df)

            # Save to history
            st.session_state['history'].append({
                "Input": user_input,
                "Prediction": prediction_label,
                "Scores": {label: round(score, 4) for label, score in zip(pred_labels, pred_scores)}
            })

            # Display history
            st.subheader("Prediction History")
            history_df = pd.DataFrame(st.session_state['history'])
            st.dataframe(history_df)

            # Bar Chart
            st.subheader("Sentiment Score Visualization")
            chart = alt.Chart(scores_df).mark_bar().encode(
                x=alt.X('Label', sort=None),
                y='Score',
                color='Label'
            ).properties(
                width=400,
                height=300
            )
            st.altair_chart(chart)

            # -------------------- SHAP Explainability --------------------
            st.subheader("Model Explanation (SHAP)")

            # Prediction wrapper for SHAP
            def predict_proba(texts):
                inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

            try:
                masker = shap.maskers.Text(tokenizer)
                explainer = shap.Explainer(predict_proba, masker)
                shap_values = explainer([user_input])
                
                st.write("Explanation below shows how each word influences the sentiment prediction:")
                shap.initjs()
                st_shap_plot = shap.plots.text(shap_values[0])
                st.pyplot(st_shap_plot.figure)
            except Exception as shap_err:
                st.warning("⚠️ SHAP explanation could not be generated. This may be due to model/tokenizer compatibility or resource limits.")

        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {e}")
            
