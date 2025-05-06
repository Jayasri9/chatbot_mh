from flask import Flask, render_template, request, jsonify
import random
import pickle
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from googletrans import Translator

app = Flask(__name__)

# Load model and related files
print("Loading model and files...")
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
intents = pickle.load(open('intents.pkl', 'rb'))

# Initialize lemmatizer and translator
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Initialize conversation context
conversation_context = {
    "last_intent": None,
    "last_language": "en",
    "repeat_count": 0
}

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

def detect_language(text):
    """Detect the language of the input text"""
    try:
        detection = translator.detect(text)
        return detection.lang
    except Exception as e:
        print(f"Language detection error: {e}")
        return "en"  # Default to English if detection fails

def translate_to_english(text, src_lang='auto'):
    """Translate text to English if it's not in English"""
    try:
        # Detect language and translate if not English
        translation = translator.translate(text, dest='en', src=src_lang)
        return translation.text, translation.src
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text, "en"

def translate_to_source_language(text, dest_lang):
    """Translate text from English to the source language"""
    try:
        if dest_lang != 'en':
            translation = translator.translate(text, dest=dest_lang, src='en')
            return translation.text
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        # Return original text if translation fails
        return text

def get_intent(text, threshold=0.3):
    """Get intent with confidence threshold"""
    # Vectorize the input text
    vec_input = vectorizer.transform([text])
    
    # Get prediction probabilities
    proba = model.predict_proba(vec_input)[0]
    max_proba = max(proba)
    
    # Check if the confidence is above threshold
    if max_proba >= threshold:
        prediction = model.predict(vec_input)[0]
        return prediction, max_proba
    else:
        return None, max_proba

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    global conversation_context
    user_input = request.form["msg"]
    
    # Detect input language
    detected_lang = detect_language(user_input)
    conversation_context["last_language"] = detected_lang
    
    # Translate to English if not in English
    translated_input, src_lang = translate_to_english(user_input, detected_lang)
    print(f"Original: '{user_input}' | Translated: '{translated_input}' | Language: {src_lang}")
    
    # Clean and preprocess text
    cleaned_input = clean_text(translated_input)
    
    # Get intent with confidence
    intent, confidence = get_intent(cleaned_input)
    print(f"Predicted intent: {intent} with confidence: {confidence}")
    
    # Check for repeated intents
    if intent == conversation_context["last_intent"]:
        conversation_context["repeat_count"] += 1
    else:
        conversation_context["repeat_count"] = 0
    
    # Handle response based on confidence and repetition
    if intent is None:
        if src_lang != 'en':
            response = translate_to_source_language(
                "I'm not sure I understand. Could you rephrase that?", src_lang)
        else:
            response = "I'm not sure I understand. Could you rephrase that?"
    else:
        # Find matching intent
        for intent_data in intents["intents"]:
            if intent_data["tag"] == intent:
                # Handle repeated responses
                if conversation_context["repeat_count"] > 1:
                    # Use a different response or ask for more detail
                    alt_responses = [
                        "I'd like to understand better. Could you share more details?",
                        "Let's explore this further. Can you tell me more?",
                        "I want to help you better. Could you elaborate on what you're experiencing?"
                    ]
                    response_text = random.choice(alt_responses)
                    conversation_context["repeat_count"] = 0  # Reset counter
                else:
                    response_text = random.choice(intent_data["responses"])
                
                # Translate response back to source language if needed
                if src_lang != 'en':
                    response = translate_to_source_language(response_text, src_lang)
                else:
                    response = response_text
                break
    
    # Update conversation context
    conversation_context["last_intent"] = intent
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)