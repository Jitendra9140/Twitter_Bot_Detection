from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pkl', 'rb'))
tweeter = pd.read_csv('bot_detection_data.csv')

# Define a function to preprocess the input data
def preprocess_input(username, tweet, hashtag, location, created_at, retweet_count, mention_count, follower_count, verified_status):
    # Convert created_at to Unix timestamp
    created_at_datetime = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
    created_at_unix = int(time.mktime(created_at_datetime.timetuple()))
    
    # Prepare input data as DataFrame
    input_data = pd.DataFrame({
        'username': [username],
        'tweet': [tweet],
        'location': [location],
        'hashtag': [hashtag],
        'retweet_count': [retweet_count],
        'verified_status': [verified_status],
        'mention_count': [mention_count],
        'follower_count': [follower_count],
        'created_at_unix': [created_at_unix]
    })

    return input_data

# Define a function to make predictions
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
    retweet_count = int(request.form.get('rtweet', 0))  # Default to 0 if not provided
    mention_count = int(request.form.get('mcount', 0))  # Default to 0 if not provided
    follower_count = int(request.form.get('fcount', 0))  # Default to 0 if not provided
    verified_status = bool(request.form.get('verified', False))  # Convert to boolean

    # Preprocess input data
    input_data = preprocess_input(username, tweet, hashtag, location, created_at, retweet_count, mention_count, follower_count, verified_status)

    # Make prediction
    prediction = make_prediction(input_data)

    # Return the prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

