import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Sentiment Analysis App")

text_input = st.text_area("Enter a sentence to analyze sentiment")

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred = model.predict([text_input])[0]
        proba = model.predict_proba([text_input]).max()
        st.write(f"**Prediction:** {pred}")
        st.write(f"**Confidence:** {proba:.2f}")
