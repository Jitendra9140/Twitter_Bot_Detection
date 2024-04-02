from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pkl', 'rb'))
tweeter = pd.read_csv('bot_detection_data.csv')
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def preprocess_input(username, tweet, hashtag, location, created_at, retweet_count, mention_count, follower_count, verified_status):

    # Prepare input data as DataFrame
    input_data = pd.DataFrame({
        'Username': [username],
        'Tweet': [tweet],
        'Location': [location],
        'Hashtags': [hashtag],
        'Retweet Count': [retweet_count],
        'Verified': [verified_status],
        'Mention Count': [mention_count],
        'Follower Count': [follower_count],
        'Created At': [created_at]
    })

    new_text_data = input_data['Tweet'] + ' ' + input_data['Username'] + \
        ' ' + input_data['Hashtags']+' '+input_data['Location']

    new_text_sparse = vectorizer.transform(new_text_data)

    new_additional_features = input_data[[
        'Retweet Count', 'Verified', 'Mention Count', 'Follower Count', 'Created At']]
    new_additional_features = new_additional_features.astype('float64')
    new_combined_sparse = hstack((new_text_sparse, new_additional_features))

    return new_combined_sparse


def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    username = request.form.get('username')
    tweet = request.form.get('Tweet')
    hashtag = request.form.get('Hashtag')
    location = request.form.get('location')
    created_at = request.form.get('created')
    # Default to 0 if not provided
    retweet_count = int(request.form.get('rtweet', 0))
    mention_count = int(request.form.get('mcount', 0))
    follower_count = int(request.form.get('fcount', 0))
    verified_status = bool(request.form.get(
        'verified', False))  # Convert to boolean

    input_data = preprocess_input(username, tweet, hashtag, location, created_at,
                                  retweet_count, mention_count, follower_count, verified_status)

    prediction = make_prediction(input_data)

    return "BOT" if prediction else "Not Bot"


if __name__ == '__main__':
    app.run(debug=True)
