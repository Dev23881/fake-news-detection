# Fake News Detection using NLP, LSTM and Streamlit

## 📌 Project Overview

Fake News Detection is a Natural Language Processing (NLP) based machine learning project that classifies news articles as **Fake** or **Real**.

The project uses text preprocessing, tokenization, padding and an **LSTM (Long Short-Term Memory)** deep learning model for news classification.

A Streamlit-based interface can be used to enter news text and display the prediction result.

## 🎯 Objectives

- Detect whether a given news article is Fake or Real.
- Apply Natural Language Processing techniques to news text.
- Train an LSTM-based deep learning model.
- Provide an easy-to-use interface for prediction.
- Display the prediction along with confidence.

## 🛠️ Technologies Used

- Python
- Natural Language Processing (NLP)
- TensorFlow / Keras
- LSTM
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Google Colab
- GitHub

## 📂 Dataset

The project uses a dataset containing fake and real news articles.

The dataset contains approximately **9,900 news records**.

The labels used in the project are:

- `0` → Fake News
- `1` → Real News

## 🔄 Methodology

The main workflow of the project is:

**News Text → Text Cleaning → Tokenization → Padding → Embedding → LSTM → Prediction**

### 1. Text Preprocessing

The news text is cleaned before training. The preprocessing includes:

- Converting text to lowercase
- Removing URLs
- Removing HTML tags
- Removing mentions and hashtags
- Removing punctuation
- Removing numbers
- Removing extra spaces

### 2. Tokenization

The cleaned text is converted into numerical sequences using a Keras Tokenizer.

- Maximum vocabulary size: `10,000`
- Out-of-vocabulary token: `<OOV>`

### 3. Padding

The sequences are padded to a fixed length.

- Maximum sequence length: `200`
- Padding: `post`
- Truncation: `post`

### 4. LSTM Model

An LSTM-based neural network is used to learn patterns from the news text and classify it as Fake or Real.

### 5. Prediction

The model produces a probability score. A threshold of `0.5` is used to classify the news:

- Probability >= 0.5 → Real News
- Probability < 0.5 → Fake News

## 📊 Model Performance

The model achieved approximately **97.12% accuracy** on the test dataset.

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Fake  | 0.98 | 0.96 | 0.97 |
| Real  | 0.96 | 0.98 | 0.97 |

## 💻 Project Structure

```text
Fake-News-Detection/
│
├── Fake_News_Detection.ipynb
├── app.py
├── fake_news_model.keras
├── tokenizer.pkl
├── README.md
└── requirements.txt
