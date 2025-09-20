import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap
import torch
import pandas as pd
import altair as alt

# -------------------- Model Setup --------------------
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Sentiment Analysis Tool", layout="centered")
st.title("Advanced Sentiment Analysis Tool")
st.write("Enter a sentence or review to analyze its sentiment using deep learning.")

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
            # Run sentiment classification
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score']

            # Display prediction
            st.success(f"**Predicted Sentiment:** {label} (Score: {score:.4f})")

            # Store result in history
            st.session_state['history'].append({
                "Input": user_input,
                "Prediction": label,
                "Score": round(score, 4)
            })

            # Display detailed scores (only POSITIVE/NEGATIVE here)
            scores_df = pd.DataFrame({
                "Label": [label],
                "Score": [score]
            })
            st.subheader("Prediction Score")
            st.dataframe(scores_df)

            # Show history
            st.subheader("Prediction History")
            history_df = pd.DataFrame(st.session_state['history'])
            st.dataframe(history_df)

            # Visualization: Bar chart
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

            # -------------------- SHAP Explanation --------------------
            st.subheader("Model Explanation (SHAP)")

            def predict_proba(texts):
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

            try:
                masker = shap.maskers.Text(tokenizer)
                explainer = shap.Explainer(predict_proba, masker)
                shap_values = explainer([user_input])

                st.write("Explanation: Words influencing the sentiment prediction")
                shap.initjs()
                st_shap_plot = shap.plots.text(shap_values[0])
                st.pyplot(st_shap_plot.figure)

            except Exception as shap_error:
                st.warning("⚠️ SHAP explanation could not be generated. It may be due to limited resources.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
