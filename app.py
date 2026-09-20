import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("fake_news_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(news_text):
    cleaned_text = clean_text(news_text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    if prediction >= 0.5:
        result = "REAL NEWS"
        confidence = prediction * 100
    else:
        result = "FAKE NEWS"
        confidence = (1 - prediction) * 100

    return result, confidence


st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or headline below to check "
    "whether the model predicts it as Fake or Real."
)

st.divider()

news_text = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste your news article or headline here..."
)

if st.button("🔍 Check News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("Please enter some news text first.")

    else:
        result, confidence = predict_news(news_text)

        st.subheader("Prediction Result")

        if result == "REAL NEWS":
            st.success(f"✅ {result}")
        else:
            st.error(f"⚠️ {result}")

        st.write(f"Confidence: **{confidence:.2f}%**")

        st.progress(int(confidence))


st.divider()

st.caption(
    "Fake News Detection using LSTM-based Natural Language Processing"
)