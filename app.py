# app.py

import streamlit as st
from transformers import pipeline

st.title("📰 Real-Time Fake News Detection")
st.write("Paste any news text below and get an instant prediction (Fake/Real) using a pretrained AI model.")

# Input
user_input = st.text_area("Enter news text here:")

# Load pretrained model (text classification)
@st.cache_resource
def load_model():
    # Use HuggingFace's 'facebook/bart-large-mnli' for zero-shot classification
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

classifier = load_model()

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter news text!")
    else:
        candidate_labels = ["fake", "real"]
        result = classifier(user_input, candidate_labels)
        label = result['labels'][0]
        score = result['scores'][0]

        if label == "real":
            st.success(f"✅ This news is predicted as *REAL* ({score*100:.2f}% confidence)")
        else:
            st.error(f"❌ This news is predicted as *FAKE* ({score*100:.2f}% confidence)")
