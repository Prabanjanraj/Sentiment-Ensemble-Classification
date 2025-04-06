import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("stacking_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("Sentiment Classification App")

# Text input
user_input = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        vect_input = vectorizer.transform([user_input])

        # Predict label
        prediction = model.predict(vect_input)[0]

        # Confidence score (max probability)
        probas = model.predict_proba(vect_input)
        confidence = np.max(probas) * 100

        # Display results
        st.markdown(f"**Prediction:** `{prediction}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")
