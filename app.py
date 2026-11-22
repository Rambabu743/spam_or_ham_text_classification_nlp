import streamlit as st
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
@st.cache_resource
def setup_nltk():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    return set(stopwords.words("english")), WordNetLemmatizer()


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("spam_pipeline.pkl")


stop_words, lemmatizer = setup_nltk()
model = load_model()


# Text preprocessing
def preprocess_message(message):
    message = message.lower()

    words = word_tokenize(
        re.sub(r"[^a-zA-Z]", " ", message)
    )

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Streamlit application
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Detection")

st.write(
    "Enter a message below to check whether it is Spam or Not Spam."
)


# User input
message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)


# Prediction button
if st.button("Check Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        try:
            # Preprocess input message
            processed_message = preprocess_message(message)

            # Predict using trained model
            prediction = model.predict([processed_message])[0]

            st.subheader("Prediction Result")

            if int(prediction) == 1:
                st.error("🚨 This message is SPAM.")

            else:
                st.success("✅ This message is NOT SPAM (HAM).")

        except Exception as e:
            st.error(f"Prediction error: {e}")