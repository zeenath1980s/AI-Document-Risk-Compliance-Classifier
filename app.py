import os
import re
import tempfile
import joblib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document

MODEL_PATH = "document_risk_model.pkl"
THRESHOLD = 70


st.set_page_config(
    page_title="AI Document Risk & Compliance Classifier",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload document_risk_model.pkl to the repository.")
    st.stop()

model = load_model()


def extract_text(file_bytes, filename):
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".txt":
        return file_bytes.decode("utf-8", errors="ignore").strip()

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        if extension == ".pdf":
            reader = PdfReader(temp_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text.strip()

        elif extension == ".docx":
            document = Document(temp_path)
            text = "\n".join(
                paragraph.text for paragraph in document.paragraphs
            )
            return text.strip()

        else:
            raise ValueError("Unsupported file type.")

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def get_important_terms(text):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    stop_words = {
        "the", "and", "may", "for", "with", "from",
        "this", "that", "company", "into", "must",
        "should", "will", "are", "was", "has", "have"
    }

    tfidf = model.named_steps["tfidf"]
    known_words = set(tfidf.get_feature_names_out())

    important_words = []

    for word in words:
        if (
            word not in stop_words
            and word in known_words
            and word not in important_words
        ):
            important_words.append(word)

    return important_words[:10]


# -------------------------------
# MAIN DASHBOARD
# -------------------------------

st.title("📄 AI Document Risk & Compliance Classifier")

st.write(
    "Upload a business document to classify its risk level "
    "and identify documents that require manual compliance review."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"]
)


if uploaded_file is not None:

    st.info(f"Selected file: {uploaded_file.name}")

    if st.button("🔍 Analyze Document", type="primary"):

        try:
            text = extract_text(
                uploaded_file.getvalue(),
                uploaded_file.name
            )

            if not text:
                st.error(
                    "No readable text was found in this document."
                )
                st.stop()

            # Prediction
            prediction = model.predict([text])[0]

            probabilities = model.predict_proba([text])[0]

            confidence = float(max(probabilities) * 100)

            # Manual review decision
            if confidence < THRESHOLD:
                status = "MANUAL REVIEW REQUIRED"
            else:
                status = "AUTOMATIC CLASSIFICATION"

            st.subheader("📊 Classification Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Risk Level",
                    prediction
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:
                st.metric(
                    "Review Threshold",
                    f"{THRESHOLD}%"
                )

            # Status
            if confidence < THRESHOLD:
                st.warning(
                    "⚠️ Manual Review Required"
                )
                st.write(
                    "The model confidence is below the required threshold."
                )
            else:
                st.success(
                    "✅ Automatic Classification"
                )

            # Probability table
            st.subheader("📈 Risk Probability")

            probability_df = pd.DataFrame({
                "Risk Level": model.classes_,
                "Probability": [
                    f"{probability * 100:.2f}%"
                    for probability in probabilities
                ]
            })

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            # Important terms
            important_terms = get_important_terms(text)

            if important_terms:
                st.subheader("🔎 Important Terms")

                st.write(
                    " • ".join(important_terms)
                )

            # Extracted text
            with st.expander("📄 View Extracted Text"):
                st.text(text)

        except Exception as e:
            st.error(
                f"Error while analyzing the document: {e}"
            )
