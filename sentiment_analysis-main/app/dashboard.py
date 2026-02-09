import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
from wordcloud import WordCloud, STOPWORDS

# -------------------------------
# Title
# -------------------------------
st.title("Analysis Dashboard")

# -------------------------------
# Load Data
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sentiment_results.csv"
SUMMARY_PATH = BASE_DIR / "data" / "summaries.json"

results = pd.read_csv(DATA_PATH)

# Normalize sentiment
results['predicted_sentiment'] = results['predicted_sentiment'].str.lower()

# -------------------------------
# Raw Data
# -------------------------------
st.subheader("Raw Data")
st.dataframe(results)

# -------------------------------
# Sentiment Distribution
# -------------------------------
st.subheader("Sentiment Distribution")
st.write(results['predicted_sentiment'].value_counts())

fig, ax = plt.subplots()
sns.countplot(x='predicted_sentiment', data=results, ax=ax)
st.pyplot(fig)

# -------------------------------
# Filter by Sentiment
# -------------------------------
st.subheader("Filter by Sentiment")
option = st.selectbox("Choose sentiment", results['predicted_sentiment'].unique())
filtered = results[results['predicted_sentiment'] == option]
st.dataframe(filtered)

# -------------------------------
# Load Summaries
# -------------------------------
with open(SUMMARY_PATH) as f:
    summaries = json.load(f)

emotion_summaries = summaries["emotion_summaries"]

# -------------------------------
# Overall Summary
# -------------------------------
st.subheader("🧾 Overall Summary")
st.info(summaries["overall_summary"])

# -------------------------------
# Sentiment-wise Summary
# -------------------------------
st.subheader("📝 Sentiment-wise Summary")

selected_sentiment = st.selectbox(
    "Select sentiment to view summary",
    options=list(emotion_summaries.keys())
)

st.success(emotion_summaries[selected_sentiment])

# =========================================================
# WORD CLOUD BY SELECTED SENTIMENT (WITH ALL OPTION)
# =========================================================
st.subheader("Word Cloud by Sentiment")

sentiment_colors = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
}

# Add ALL option
wc_options = ["All"] + list(results['predicted_sentiment'].unique())

selected_wc_sentiment = st.selectbox(
    "Select sentiment for WordCloud",
    wc_options
)

# Prepare text
if selected_wc_sentiment == "All":
    text_data = " ".join(
        results["comment_text"].dropna().astype(str)
    )
    color = None
else:
    text_data = " ".join(
        results[results['predicted_sentiment'] == selected_wc_sentiment]
        ["comment_text"].dropna().astype(str)
    )
    color = sentiment_colors.get(selected_wc_sentiment, "blue")

# Generate wordcloud
if text_data.strip():

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(text_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    if color:
        wordcloud = wordcloud.recolor(color_func=lambda *args, **kwargs: color)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

else:
    st.warning("No comments available for this sentiment.")
