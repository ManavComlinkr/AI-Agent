from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
from fuzzywuzzy import fuzz
from waitress import serve

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin support

# Load and merge two datasets
datasets = [
    pd.read_csv("xcavate_master_chatbot_dataset.csv"),
    pd.read_csv("realxmarket_chatbot_intents_full.csv")
]
chat_data = pd.concat(datasets, ignore_index=True)

# Create a dictionary to group responses by intent
intent_responses = {}
for _, row in chat_data.iterrows():
    intent = row['Intent']
    if intent not in intent_responses:
        intent_responses[intent] = []
    intent_responses[intent].append((row['User Example'], row['Bot Response']))

# Improved intent detection using fuzzy matching
def detect_intent(user_input, threshold=70):
    user_input = user_input.lower()
    best_match = ("fallback", 0)
    for intent, examples in intent_responses.items():
        for example, _ in examples:
            score = fuzz.partial_ratio(example.lower(), user_input)
            if score > best_match[1]:
                best_match = (intent, score)
    return best_match[0] if best_match[1] >= threshold else "fallback"

# Get a response based on detected intent
def get_response(intent):
    if intent in intent_responses:
        return random.choice(intent_responses[intent])[1]
    else:
        return "I'm not sure how to respond to that yet. Could you rephrase or try something else?"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    intent = detect_intent(user_input)
    response = get_response(intent)
    return jsonify({"intent": intent, "response": response})

if __name__ == "__main__":
    print("Starting Xcavate Chatbot API in Production Mode...")
    serve(app, host="0.0.0.0", port=5000)
