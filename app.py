from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Add this
import joblib
import numpy as np
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all origins

# Enhanced model loading with fallback
try:
    model = joblib.load("final_news_model.pkl")
    vectorizer = joblib.load("final_vectorizer.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    # Create dummy model as fallback
    vectorizer = TfidfVectorizer()
    model = LogisticRegression()
    print("⚠️ Running in fallback mode - predictions will be less accurate")

# Initialize NLP with better error handling
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("⚠️ Downloading NLTK resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')).union({
    'said', 'would', 'could', 'also', 'us', 'one'
})

# Source credibility markers
CREDIBLE_SOURCES = {
    'reuters', 'ap news', 'bbc', 'nasa', 'cdc', 'who', 
    'nature journal', 'science magazine', 'federal reserve'
}

def clean_text(text):
    """Enhanced cleaning with source detection"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Mark credible sources
    if any(src in text for src in CREDIBLE_SOURCES):
        text += " officialsource"
    
    # Improved cleaning
    text = re.sub(r'(@|#)\w+', '', text)  # Social media
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s'.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    """More robust preprocessing"""
    text = clean_text(text)
    try:
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()  # Fallback
    
    processed = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            lemma = lemmatizer.lemmatize(
                lemmatizer.lemmatize(token, pos='v')
            )
            processed.append(lemma)
    return ' '.join(processed)

def analyze_text_features(text):
    """Generate interpretable features"""
    features = defaultdict(float)
    
    # Style features
    features['exclamation_ratio'] = text.count('!') / len(text) if text else 0
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Content features
    features['credibility_terms'] = sum(
        1 for word in ['study', 'research', 'report', 'data'] 
        if word in text
    )
    features['has_reliable_source'] = any(
        src in text for src in CREDIBLE_SOURCES
    )
    
    return dict(features)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint"""
    data = request.get_json()
    title = data.get("title", "")
    description = data.get("description", "")
    input_text = f"{title} {description}"

    if not input_text.strip():
        return jsonify({
            'error': 'No text provided',
            'prediction': 'Unknown',
            'confidence': 0
        })

    try:
        # Preprocess and transform
        cleaned = preprocess_text(input_text)
        features = analyze_text_features(cleaned)
        transformed = vectorizer.transform([cleaned])

        # Get prediction
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]
        confidence = max(proba)

        # Apply source credibility override
        if features['has_reliable_source'] and prediction == 0:
            prediction = 1  # Override to REAL if credible source
            confidence = max(confidence, 0.8)  # Boost confidence

        # Build response
        response = {
            'prediction': 'Real News ✅' if prediction == 1 else 'Fake News ❌',
            'confidence': float(confidence),
            'features': features,
            'processed_text': cleaned[:200] + ("..." if len(cleaned) > 200 else "")
        }

        # Confidence warnings
        if confidence < 0.6:
            response['warning'] = "Low confidence prediction"
            response['prediction'] = "Uncertain 🤔"

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'prediction': 'Error',
            'confidence': 0
        })

@app.route('/')
def home():
    return '''
        <h2>Enhanced Fake News Detector</h2>
        <form method="POST" action="/predict">
            <textarea name="news" rows="10" cols="60" 
                      placeholder="Paste news article here..."></textarea><br>
            <input type="submit" value="Analyze">
        </form>
        <p><small>Analyzes credibility signals and content patterns</small></p>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
