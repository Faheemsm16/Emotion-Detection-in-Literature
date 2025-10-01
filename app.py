# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

# -----------------------------
# STOPWORDS (Python-only, avoids NLTK)
STOPWORDS = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","yourselves","he","him","his","himself","she","her",
    "hers","herself","it","its","itself","they","them","their","theirs",
    "themselves","what","which","who","whom","this","that","these","those",
    "am","is","are","was","were","be","been","being","have","has","had","having",
    "do","does","did","doing","a","an","the","and","but","if","or","because","as",
    "until","while","of","at","by","for","with","about","against","between",
    "into","through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","again","further","then","once",
    "here","there","when","where","why","how","all","any","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own","same",
    "so","than","too","very","s","t","can","will","just","don","should","now"
])

# -----------------------------
# STREAMLIT UI
st.set_page_config(page_title="🎭 Poem Emotion Analyzer", layout="wide")
st.title("🎭 Poem Emotion Analyzer")

poem = st.text_area("Enter your poem here:")

# -----------------------------
# LOAD TRANSFORMER MODEL (cached)
@st.cache_resource
def load_model():
    model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Text preprocessing
def preprocess_text(text):
    words = [w for w in text.lower().split() if w.isalnum() and w not in STOPWORDS]
    return " ".join(words)

# Sliding window
def sliding_window(text, window_size=20, step_size=10):
    words = text.split()
    return [" ".join(words[i:i+window_size]) for i in range(0, len(words), step_size)]

# Emotion analysis
EMOTION_LABELS = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                  "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                  "excitement","fear","gratitude","grief","joy","love","nervousness","neutral",
                  "optimism","pride","realization","relief","remorse","sadness","surprise"]

def analyze_emotions(poem, threshold=0.065):
    inputs = tokenizer(poem, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    emotion_scores = {EMOTION_LABELS[i]: scores[i] for i in range(len(EMOTION_LABELS)) if scores[i] >= threshold}
    sorted_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:6])
    return sorted_emotions

# -----------------------------
# PLOTTING
def plot_emotion_heatmap(emotion_trends):
    emotions = list(emotion_trends.keys())
    max_windows = max(len(scores) for scores in emotion_trends.values())
    emotion_matrix = np.zeros((len(emotions), max_windows))
    for i, emotion in enumerate(emotions):
        scores = emotion_trends[emotion]
        emotion_matrix[i, :len(scores)] = scores
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(emotion_matrix, annot=True,
                xticklabels=[f"Win {i+1}" for i in range(max_windows)],
                yticklabels=emotions, cmap='Blues', ax=ax)
    ax.set_title("Emotion Heatmap Across Sliding Windows")
    ax.set_xlabel("Sliding Windows")
    ax.set_ylabel("Emotions")
    return fig

def plot_emotion_flow(emotion_trends):
    fig, ax = plt.subplots(figsize=(12,6))
    for emotion, scores in emotion_trends.items():
        ax.plot(range(1, len(scores)+1), scores, marker='o', label=emotion)
    ax.set_title("Emotion Flow Across Sliding Windows")
    ax.set_xlabel("Sliding Windows")
    ax.set_ylabel("Emotion Score")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1))
    ax.grid(True)
    return fig

# -----------------------------
# RUN ANALYSIS
if st.button("Analyze"):
    if not poem.strip():
        st.warning("Please enter some text.")
    else:
        processed_poem = preprocess_text(poem)
        windows = sliding_window(processed_poem)
        emotion_trends = defaultdict(list)
        for window in windows:
            emotions = analyze_emotions(window)
            for emotion, score in emotions.items():
                emotion_trends[emotion].append(score)

        st.subheader("Transformer-Based Emotion Analysis (Average Scores)")
        sorted_emotions = sorted(
            [(e, sum(s)/len(s)) for e,s in emotion_trends.items()],
            key=lambda x: x[1], reverse=True
        )
        for emotion, avg_score in sorted_emotions:
            st.write(f"**{emotion.capitalize()}**: {avg_score:.2f}")

        st.subheader("Emotion Heatmap")
        st.pyplot(plot_emotion_heatmap(emotion_trends))

        st.subheader("Emotion Flow Across Windows")
        st.pyplot(plot_emotion_flow(emotion_trends))
