import streamlit as st
from transformers import pipeline

# Cache model
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", use_fast=False)

classifier = load_model()

st.title("Klasifikacija Upita (Zero-Shot)")

# User input
user_text = text = st.text_input(
        "Please enter a term to classify.",
        value="",
        max_chars=100,
        key="query",
    )

if st.button("Klasificiraj"):
    if user_text.strip():
        labels = ["problem prijave", "reset lozinke", "cijena paketa", "ostalo"]
        result = classifier(user_text, candidate_labels=labels)
        st.write("Najvjerojatnija kategorija:", result['labels'][0])
        st.json(result)
    else:
        st.warning("Molimo unesite tekst za klasifikaciju.")
