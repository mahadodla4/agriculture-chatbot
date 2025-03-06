import nltk
import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import warnings
warnings.filterwarnings('ignore')
import pickle
from model import NeuralNetwork  # Import the model class from the training file

stemmer = WordNetLemmatizer()
# Load words and classes
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load intents
with open('intents_big.json') as file:
    intents = json.load(file)

# Load the trained model
input_size = len(words)
hidden_size = 8
output_size = len(classes)
model = NeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('data.pth'))
model.eval()

# Function to preprocess the input sentence
def preprocess_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [stemmer.lemmatize(word) for word in sentence_words if word in words]
    return sentence_words

# Function to convert the preprocessed sentence into a feature vector
def sentence_to_features(sentence_words):
    features = [1 if word in sentence_words else 0 for word in words]
    return torch.tensor(features).float().unsqueeze(0)

# Function to generate a response
def generate_response(sentence):
    sentence_words = preprocess_sentence(sentence)
    if len(sentence_words) == 0:
        return "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"

    features = sentence_to_features(sentence_words)
    with torch.no_grad():
        outputs = model(features)

    probabilities, predicted_class = torch.max(outputs, dim=1)
    confidence = probabilities.item()
    predicted_tag = classes[predicted_class.item()]

    if confidence > 0.5:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['response'])

    return "I'm sorry, but I'm not sure how to respond to that."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break
        resp = generate_response(sentence)
        print(resp)