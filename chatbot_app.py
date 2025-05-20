# chatbot_app.py
from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
import numpy as np
import pickle
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# Load data
with open('intents.json') as file:
    intents = json.load(file)

# Load trained model and vectorizer
model = pickle.load(open('chatbot_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Flask app setup
app = Flask(__name__)

# Function: Text preprocessing
def preprocess(text):
    text = correct_spelling(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function: Spelling correction
def correct_spelling(text):
    return str(TextBlob(text))

# Function: Predict tag from input text
def predict_class(text):
    text = preprocess(text)
    X = vectorizer.transform([text])
    tag = model.predict(X)[0]
    return tag

# Function: Fallback token-overlap-based matching
def fallback_match(text):
    text = preprocess(text)
    text_tokens = set(text.split())
    best_tag = None
    best_score = 0.0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = set(preprocess(pattern).split())
            common_tokens = text_tokens.intersection(pattern_tokens)
            score = len(common_tokens) / max(len(pattern_tokens), 1)
            if score > best_score:
                best_score = score
                best_tag = intent['tag']

    if best_score > 0.2:
        return best_tag
    return None

# Function: Generate bot response
def get_response(text):
    try:
        intent_tag = predict_class(text)
    except:
        intent_tag = None

    if intent_tag:
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])

    fallback_tag = fallback_match(text)
    if fallback_tag:
        for intent in intents['intents']:
            if intent['tag'] == fallback_tag:
                return random.choice(intent['responses'])

    return "I'm not sure I understand. Could you rephrase that?"

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_message = request.form["msg"]
    response = get_response(user_message)
    return response

if __name__ == "__main__":
    app.run(debug=True)
