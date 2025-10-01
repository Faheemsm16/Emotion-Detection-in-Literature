import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import pandas as pd
from nltk.tokenize import word_tokenize

# Download resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load model once
@st.cache_resource
def load_model():
    model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Preprocessing
def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(words)

def sliding_window(text, window_size=20, step_size=10):
    words = text.split()
    return [" ".join(words[i:i+window_size]) for i in range(0, len(words), step_size)]

# Transformer analysis
def analyze_emotions(poem, threshold=0.065):
    inputs = tokenizer(poem, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze().tolist()

    emotion_labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                      "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                      "gratitude", "grief", "joy", "love", "nervousness", "neutral", "optimism", "pride", "realization",
                      "relief", "remorse", "sadness", "surprise"]

    emotion_scores = {emotion_labels[i]: scores[i] for i in range(len(emotion_labels)) if scores[i] >= threshold}
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:6]
    return dict(sorted_emotions)

# Plot functions
def plot_emotion_heatmap(emotion_trends):
    emotions = list(emotion_trends.keys())
    max_windows_count = max(len(scores) for scores in emotion_trends.values())
    emotion_matrix = np.zeros((len(emotions), max_windows_count))

    for i, emotion in enumerate(emotions):
        scores = emotion_trends[emotion]
        emotion_matrix[i, :len(scores)] = scores

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(emotion_matrix, annot=True,
                xticklabels=[f"Win {i+1}" for i in range(max_windows_count)],
                yticklabels=emotions, cmap='Blues', ax=ax)
    ax.set_title("Emotion Heatmap")
    return fig

def plot_emotion_flow(emotion_trends):
    fig, ax = plt.subplots(figsize=(12, 6))
    for emotion, scores in emotion_trends.items():
        ax.plot(range(1, len(scores) + 1), scores, label=emotion, marker='o')
    ax.set_title("Emotion Flow Across Windows")
    ax.legend()
    return fig

# --- Streamlit UI ---
st.title("🎭 Poem Emotion Analyzer")
poem = st.text_area("Enter your poem:")

if st.button("Analyze"):
    processed_poem = preprocess_text(poem)
    windows = sliding_window(processed_poem)
    emotion_trends = defaultdict(list)

    for window in windows:
        emotions = analyze_emotions(window)
        for emotion, score in emotions.items():
            emotion_trends[emotion].append(score)

    st.subheader("Transformer-Based Emotion Analysis")
    sorted_emotions = sorted(
        [(e, sum(s)/len(s)) for e, s in emotion_trends.items()],
        key=lambda x: x[1], reverse=True
    )
    for emotion, avg_score in sorted_emotions:
        st.write(f"**{emotion.capitalize()}**: {avg_score:.2f}")

    st.pyplot(plot_emotion_heatmap(emotion_trends))
    st.pyplot(plot_emotion_flow(emotion_trends))
