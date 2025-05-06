import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import re

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and translator
lemmatizer = WordNetLemmatizer()
translator = Translator()

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a string
    return ' '.join(lemmatized_tokens)

def translate_to_english(text, src_lang='auto'):
    """Translate text to English if it's not in English"""
    try:
        # Detect language and translate if not English
        translation = translator.translate(text, dest='en', src=src_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text

def translate_to_source_language(text, dest_lang):
    """Translate text from English to the source language"""
    try:
        translation = translator.translate(text, dest=dest_lang, src='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text

# Load intents
print("Loading intents...")
with open('intents.json') as file:
    data = json.load(file)

# Prepare training data
print("Preparing training data...")
corpus = []
raw_corpus = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        raw_corpus.append(pattern)
        # Clean and lemmatize each pattern
        cleaned_pattern = clean_text(pattern)
        corpus.append(cleaned_pattern)
        labels.append(intent['tag'])

# Create a more advanced vectorizer with TF-IDF
print("Creating vectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(corpus)
y = labels

# Train the model
print("Training model...")
model = MultinomialNB(alpha=0.1)
model.fit(X, y)

# Evaluate model
from sklearn.model_selection import cross_val_score
print("Evaluating model...")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean()}")

# Save model and vectorizer
print("Saving model and related files...")
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(data, open('intents.pkl', 'wb'))

# Also save raw corpus for reference
pickle.dump(raw_corpus, open('raw_corpus.pkl', 'wb'))

print("Model training complete!") 