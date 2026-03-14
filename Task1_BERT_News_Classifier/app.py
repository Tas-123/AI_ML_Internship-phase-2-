# app.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
model_path = "bert_news_model"  # folder with trained model

if not os.path.exists(model_path):
    st.error("BERT model folder not found! Make sure 'bert_news_model' is in the same folder as app.py.")
else:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="News Headline Classifier", layout="centered")

st.title("📰 BERT News Headline Classifier")
st.write(
    "Enter a news headline below and get its predicted category. "
    "This model classifies headlines into World, Sports, Business, or Sci/Tech."
)

user_input = st.text_area("Enter News Headline", height=80)

# -----------------------------
# Prediction
# -----------------------------
if user_input:
    with st.spinner("Classifying..."):
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"Predicted Category: **{label_names[pred]}**")