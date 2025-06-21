from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def load_data():
    # Load the full dataset to access label names
    full_dataset = load_dataset("go_emotions", "simplified")
    # Correct way to access label names for Sequence(ClassLabel)
    label_names = full_dataset["train"].features["labels"].feature.names
    # Now get the train split
    dataset = full_dataset["train"]
    texts = dataset["text"]
    labels = dataset["labels"]
    # For multi-label, take the first label as primary (or use another strategy)
    label_strs = [label_names[l[0]] if l else "neutral" for l in labels]
    return pd.DataFrame({'text': texts, 'label': label_strs})

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', SVC(kernel='linear', probability=True))
    ])

    print("Training SVM model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "../models/svm_goemotions.pkl")
    print("Model saved to models/svm_goemotions.pkl")

if __name__ == "__main__":
    main()
