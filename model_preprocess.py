import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def load_and_prepare_data():
    fake_df = pd.read_csv("Fake.csv", encoding='utf-8')
    real_df = pd.read_csv("True.csv", encoding='utf-8')
    
    fake_df.drop_duplicates(subset=['title', 'text'], inplace=True)
    real_df.drop_duplicates(subset=['title', 'text'], inplace=True)

    # Label the datasets
    fake_df['label'] = 0
    real_df['label'] = 1

    # Combine and shuffle
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df['content'] = df['title'] + " " + df['text']
    df = shuffle(df, random_state=42)
    
    return df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    source_markers = {
        'nasa': 'nasasource',
        'cdc': 'cdcsource',
        'who': 'whosource',
        'federal reserve': 'fedsource',
        'nature': 'naturesource',
        'published': 'peerreviewed'
    }
    
    text_lower = text.lower()
    for keyword, marker in source_markers.items():
        if keyword in text_lower:
            text += f" {marker}"

    text = text.lower()
    text = re.sub(r"[^a-z\s'.,!?]", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english')).union({
        'said', 'would', 'could', 'also', 'us', 'one'
    })
    lemmatizer = WordNetLemmatizer()

    processed = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos='v'))
        for token in tokens if token not in stop_words and len(token) > 2
    ]
    return ' '.join(processed)

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(
        max_df=0.5,
        min_df=10,
        ngram_range=(1, 2),
        max_features=8000,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)

    base_model = LogisticRegression(
        class_weight={0: 1, 1: 1.5},
        max_iter=2000,
        C=0.8,
        solver='liblinear'
    )
    
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer

def main():
    print("📦 Downloading NLTK resources...")
    download_nltk_resources()

    print("\n📊 Loading and preprocessing data...")
    df = load_and_prepare_data()
    df['processed'] = df['content'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )

    print("\n🚀 Training model...")
    model, vectorizer = train_model(X_train, y_train)

    print("\n✅ Evaluating model...")
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\n✅ Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    print("\n💾 Saving model files...")
    joblib.dump(model, "final_news_model.pkl")
    joblib.dump(vectorizer, "final_vectorizer.pkl")
    print("✅ Model and vectorizer saved.")

if __name__ == "__main__":
    main()
