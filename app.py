import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils import clean_text

st.set_page_config(page_title="Real-Time Fake News Detector", layout="centered")
st.title("📰 Real-Time Fake News Detection")
st.write("Paste any news article or headline below, and the AI will classify it as Fake or Real.")

user_input = st.text_area("Enter news text here:")

@st.cache_resource
def load_model():
    model_name = "mrm8488/bert-tiny-finetuned-fake-news"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

classifier = load_model()

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter news text!")
    else:
        cleaned_text = clean_text(user_input)
        result = classifier(cleaned_text)[0]  
        label = result['label']
        score = result['score']

        if label.lower() == "real":
            st.success(f"✅ This news is predicted as *REAL* ({score*100:.2f}% confidence)")
        else:
            st.error(f"❌ This news is predicted as *FAKE* ({score*100:.2f}% confidence)")
