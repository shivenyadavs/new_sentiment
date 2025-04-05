import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import streamlit as st

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.load_state_dict(torch.load("distilbert_sentiment_model.pt", map_location=torch.device("cpu")))
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

st.title("Sentiment Analysis with DistilBERT")

user_input = st.text_area("Enter a sentence:")
if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    labels = ['Negative', 'Neutral', 'Positive']
    st.write(f"**Sentiment:** {labels[pred]}")
    st.bar_chart(probs.detach().numpy()[0])
