# app.py
import streamlit as st
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud

# Load artifacts
vect = pickle.load(open("models/tfidf.pkl","rb"))
clf  = pickle.load(open("models/clf.pkl","rb"))
sia  = SentimentIntensityAnalyzer()

st.title("🚀 Crypto Tweets Sentiment Dashboard")

# Sidebar: live prediction
st.sidebar.header("Analyze Your Own Tweet")
user_txt = st.sidebar.text_area("Paste tweet text here")
if st.sidebar.button("Predict"):
    s = sia.polarity_scores(user_txt)["compound"]
    p = clf.predict(vect.transform([user_txt]))[0]
    st.sidebar.markdown(f"**VADER compound:** {s:.3f}")
    st.sidebar.markdown(f"**Model predicts:** {p}")

# Main: load sampled data
@st.cache_data
def load_data(n=200_000):
    return pd.read_csv("data/sampled_labeled.csv", nrows=n)

df = load_data()

# 1) Sentiment distribution
fig1 = px.histogram(df, x="sentiment", title="Sentiment Distribution")
st.plotly_chart(fig1, use_container_width=True)

# 2) Word Clouds
for lbl in ["positive", "negative", "neutral"]:
    txt = " ".join(df[df.sentiment == lbl].clean_text.dropna())
    wc = WordCloud(width=400, height=200).generate(txt)
    st.subheader(f"{lbl.title()} Word Cloud")
    st.image(wc.to_array(), use_container_width=True)


