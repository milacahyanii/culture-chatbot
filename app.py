import random
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import re

from flask import Flask, request, jsonify
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Uncomment to download stopwords
# nltk.download("stopwords")

app = Flask(__name__)

# Function to clean and preprocess text
def clean_input(text):
    # Remove punctuation
    text = re.sub("[^a-zA-Z]", ' ', text)
    # Convert to lowercase
    text = text.lower().split()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words]
    # Join words back to a single string
    text = " ".join(text)
    return text

# Function to convert JSON data to DataFrame and response dictionary
def to_df(data):
    tags = []
    inputs = []
    responses = {}
    
    # Extract patterns and responses
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']
        for pattern in intent['patterns']:
            pattern = clean_input(pattern)
            inputs.append(pattern)
            tags.append(intent['tag'])

    # Create a DataFrame from the cleaned inputs and tags
    data = pd.DataFrame({"inputs": inputs, "tags": tags})
    return data, responses

# Function to fit a tokenizer
def fit_tokenizer(data, oov_token):
    tokenizer = Tokenizer(num_words=1000, oov_token=oov_token)
    tokenizer.fit_on_texts(data)
    return tokenizer

# Function to tokenize and pad sequences
def tok_pad_seq(text_pred, tokenizer):
    pred_input = tokenizer.texts_to_sequences(text_pred)
    pred_input = np.array(pred_input).reshape(-1)
    pred_input = pad_sequences(
        [pred_input], maxlen=11, padding='post', truncating='post')
    return pred_input

@app.route('/', methods=['POST'])
def index():
    text_pred = []

    # Request input text
    json_data = request.json
    pred_input = json_data['text']

    # Load intent data from JSON
    try:
        with open('intents.json', encoding='utf-8') as content:
            data = json.load(content)
    except FileNotFoundError:
        return jsonify({"error": "content.json file not found."}), 404

    # Convert JSON data to DataFrame and responses dictionary
    content_data = to_df(data)
    data = content_data[0]
    responses = content_data[1]

    # Encode intent labels
    label_encoder = preprocessing.LabelEncoder()
    labels = np.array(data['tags'])
    labels = label_encoder.fit_transform(labels)

    # Tokenize intent data
    tokenizer = fit_tokenizer(data['inputs'], "<OOV>")

    # Clean and tokenize input text
    pred_input = clean_input(pred_input)
    text_pred.append(pred_input)
    pred_input = tok_pad_seq(text_pred, tokenizer)

    # Predict the output intent tag
    try:
        output = model.predict(pred_input)
        output = output.argmax()
        tag = label_encoder.inverse_transform([output])[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Respond with a random message from the matched tag's responses
    if tag in responses:
        return jsonify(
            tag=tag,
            message=random.choice(responses[tag])
        )
    else:
        return jsonify({"message": "I'm not sure how to respond to that."}), 404

# Uncomment this to develop locally
if __name__ == '__main__':
    app.run(port=5000, debug=True)
