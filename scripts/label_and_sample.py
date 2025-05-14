# scripts/label_and_sample.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT = r"D:\crypto_sentiment_app\data\cleaned_crypto_tweets.csv"
OUTPUT = r"D:\crypto_sentiment_app\data\sampled_labeled.csv"
# Number of rows to sample from the labeled data "
SAMPLE_SIZE = 1_000_000

sia = SentimentIntensityAnalyzer()
def vader_label(text):
    if not isinstance(text, str) or not text.strip():
        return "neutral"
    c = sia.polarity_scores(text)["compound"]
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"


def main():
    # Read in chunks, label, sample
    reader = pd.read_csv(INPUT, usecols=["clean_text"], chunksize=200_000)
    labeled_chunks = []
    for chunk in reader:
        chunk["sentiment"] = chunk["clean_text"].map(vader_label)
        labeled_chunks.append(chunk.sample(frac=0.2, random_state=42))
    df = pd.concat(labeled_chunks, ignore_index=True)
    # If still too big, downsample to exact SAMPLE_SIZE
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    df.to_csv(OUTPUT, index=False)
    print("Wrote", OUTPUT, "with", len(df), "rows")

if __name__ == "__main__":
    main()
