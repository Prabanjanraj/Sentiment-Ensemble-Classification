import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load model and vectorizer
model = joblib.load("stacking_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# App title
st.title("🧠 Sentiment Classifier App")
st.write("Enter a sentence and let the ensemble model classify its sentiment!")

# Text input
user_input = st.text_area("📝 Enter your text here:", height=150)

# Optional ROC checkbox
show_roc = st.checkbox("Show ROC Curve")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        transformed = vectorizer.transform([user_input])
        pred = model.predict(transformed)[0]
        prob = model.predict_proba(transformed)[0]

        # Result
        sentiment = "Positive 😊" if pred == 1 else "Negative 😠"
        confidence = round(np.max(prob) * 100, 2)

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence}%")

        # ROC curve
        if show_roc:
            st.subheader("📈 ROC Curve (Demo with Sample Points)")

            # Simulated test data (optional: replace with real X_test, y_test)
            try:
                # Load test data if available (optional)
                X_test = joblib.load("X_test_vectorized.pkl")
                y_test = joblib.load("y_test.pkl")
                y_probs = model.predict_proba(X_test)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Receiver Operating Characteristic")
                ax.legend(loc="lower right")
                st.pyplot(fig)

            except:
                st.warning("ROC Curve demo data not found. Upload `X_test_vectorized.pkl` and `y_test.pkl` to enable it.")

