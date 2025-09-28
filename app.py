import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

# --- NLTK downloads ---
nltk.download('stopwords')
nltk.download('punkt')

# --- Load model ---
@st.cache_resource
def load_model():
    model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- Load NRC lexicon ---
@st.cache_data
def load_nrc_lexicon():
    lexicon_df = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep="\t",
                             names=["word","emotion","association"], engine='python')
    lexicon_df["association"] = pd.to_numeric(lexicon_df["association"], errors='coerce').fillna(0).astype(int)
    lexicon_df = lexicon_df.pivot(index="word", columns="emotion", values="association").fillna(0)
    return lexicon_df

nrc_lexicon = load_nrc_lexicon()
stop_words = set(stopwords.words('english'))

# --- Helper functions ---
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in stop_words]
    return " ".join(words)

def sliding_window(text, window_size=20, step_size=10):
    words = text.split()
    return [" ".join(words[i:i+window_size]) for i in range(0, len(words), step_size)]

def analyze_emotions(text, threshold=0.065):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    emotion_labels = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                      "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                      "excitement","fear","gratitude","grief","joy","love","nervousness","neutral",
                      "optimism","pride","realization","relief","remorse","sadness","surprise"]
    emotion_scores = {emotion_labels[i]: scores[i] for i in range(len(scores)) if scores[i] >= threshold}
    return dict(sorted(emotion_scores.items(), key=lambda x:x[1], reverse=True)[:6])

def lexicon_analysis(text):
    words = word_tokenize(text)
    emotions = nrc_lexicon.loc[nrc_lexicon.index.intersection(words)].sum()
    return emotions.idxmax() if emotions.sum() > 0 else "Neutral"

def plot_heatmap(emotion_trends):
    emotions = list(emotion_trends.keys())
    max_windows = max(len(scores) for scores in emotion_trends.values())
    matrix = np.zeros((len(emotions), max_windows))
    for i, e in enumerate(emotions):
        scores = emotion_trends[e]
        matrix[i, :len(scores)] = scores

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(matrix, annot=True, xticklabels=[f"W{i+1}" for i in range(max_windows)],
                yticklabels=emotions, cmap="Blues", cbar_kws={'label':'Emotion Score'}, ax=ax)
    ax.set_title("Emotion Heatmap")
    return fig

def plot_flow(emotion_trends):
    fig, ax = plt.subplots(figsize=(10,6))
    for emotion, scores in emotion_trends.items():
        ax.plot(range(1,len(scores)+1), scores, label=emotion, marker='o')
    ax.set_xlabel("Sliding Windows")
    ax.set_ylabel("Score")
    ax.set_title("Emotion Flow")
    ax.legend(bbox_to_anchor=(1.05,1))
    ax.grid(True)
    return fig

# --- Streamlit interface ---
st.title("📖 Emotion Detection in Literature")
st.write("Analyze emotions in poems or text using a fine-tuned DistilBERT model and NRC lexicon.")

poem_input = st.text_area("Enter your poem or text:", height=200)

if st.button("Analyze"):
    if poem_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        processed = preprocess_text(poem_input)
        windows = sliding_window(processed)
        emotion_trends = defaultdict(list)
        
        for window in windows:
            scores = analyze_emotions(window)
            for e,s in scores.items():
                emotion_trends[e].append(s)
        
        # Transformer-based
        top_transformer = {e: sum(s)/len(s) for e,s in emotion_trends.items()}
        top_transformer = dict(sorted(top_transformer.items(), key=lambda x:x[1], reverse=True))
        st.subheader("Transformer-Based Emotions")
        st.write(top_transformer)
        
        # Lexicon-based
        lex_result = lexicon_analysis(processed)
        st.subheader("Lexicon-Based Emotion")
        st.write(lex_result)
        
        # Plots
        st.subheader("Emotion Heatmap")
        st.pyplot(plot_heatmap(emotion_trends))
        st.subheader("Emotion Flow Across Windows")
        st.pyplot(plot_flow(emotion_trends))
