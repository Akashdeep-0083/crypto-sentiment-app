import pandas as pd

# Path to your dataset
DATA_PATH = "data/sampled_labeled.csv"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop any missing tweets
    df = df.dropna(subset=["clean_text"])

    # Sample 10 random tweets
    sample = df.sample(n=10, random_state=42)

    # Print each tweet with sentiment
    for i, row in sample.iterrows():
        print(f"{i+1}. [{row['sentiment'].upper()}] {row['clean_text']}\n")

if __name__ == "__main__":
    main()
