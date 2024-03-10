import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import gensim.downloader as api

def clean_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub(r'&[a-z]+;', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    return tweet

word_vectors = api.load("glove-twitter-100")

def vectorize_tweet(tweet, word_vectors):
    vector = np.zeros(word_vectors.vector_size)
    num_words = 0
    for word in tweet.split():
        if word in word_vectors:
            vector += word_vectors[word]
            num_words += 1
    if num_words > 0:
        vector /= num_words
    return vector

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Missing key: text'}), 400

    tweet = data['text']
    cleaned_tweet = clean_tweet(tweet)
    vectorized_tweet = vectorize_tweet(cleaned_tweet, word_vectors)
    vectorized_tweet = vectorized_tweet.reshape(1, -1)  # Reshape for single prediction
    prediction = model.predict(vectorized_tweet)
    probability = model.predict_proba(vectorized_tweet)[0][1]  # Probability of the positive class
    if probability >= 0.5:
        return jsonify({'prediction': 'Positive', 'probability': probability})
    else:
        return jsonify({'prediction': 'Negative', 'probability': 1 - probability})

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = list(request.form.values())[0]
        text_to_classify = clean_tweet(data)
        vectorized_tweet = vectorize_tweet(text_to_classify, word_vectors)
        vectorized_tweet = vectorized_tweet.reshape(1, -1)  # Reshape for single prediction
        prediction = model.predict(vectorized_tweet)
        if prediction == 0:
            return render_template("predict.html", prediction = 'Negative')
        else:
            return render_template("predict.html", prediction = 'Positive')
    else:
        return render_template("form.html")
    
    


if __name__ == '__main__':
    app.run(debug = True)