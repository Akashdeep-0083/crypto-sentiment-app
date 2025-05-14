# scripts/train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Use forward slashes or raw strings on Windows
DATA     = r"data\sampled_labeled.csv"
VECT_OUT = r"models\tfidf.pkl"
CLF_OUT  = r"models\clf.pkl"

def main():
    # 1) Load & drop any rows missing text/labels
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["clean_text", "sentiment"])
    
    # 2) Split
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"], df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    # 3) Vectorize
    vect = TfidfVectorizer(max_features=50_000, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train)
    Xv  = vect.transform(X_val)

    # 4) Train
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)

    # 5) Evaluate
    print(classification_report(y_val, clf.predict(Xv)))

    # 6) Save artifacts
    pickle.dump(vect, open(VECT_OUT, "wb"))
    pickle.dump(clf, open(CLF_OUT,  "wb"))
    print("✅ Models saved to", VECT_OUT, "and", CLF_OUT)

if __name__ == "__main__":
    main()
