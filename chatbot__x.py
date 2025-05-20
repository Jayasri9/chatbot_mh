import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load intents
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Load trained model
data = torch.load("data.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Pandora"
confidence_threshold = 0.75

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() >= confidence_threshold:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    else:
        # Check if 'no-response' or 'default' tags exist for fallback
        fallback_tags = ["no-response", "default"]
        for fallback in fallback_tags:
            for intent in intents["intents"]:
                if intent["tag"] == fallback:
                    return random.choice(intent["responses"])
        # Final fallback
        return