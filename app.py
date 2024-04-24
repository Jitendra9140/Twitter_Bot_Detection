from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd


app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pkl', 'rb'))


def preprocess_input(Friends, Followers, Url, total_tweets, User_mention):
    inputData = {
        "Friends": Friends,
        "Followers": Followers,
        "Url": Url,
        "total_tweets": total_tweets,
        "User_mention": User_mention,
    }
# rf = followers / (followers + friends )    
# Url = Url/Total_tweets
# User_mentions = User_mentions/ Total_tweets
    df = pd.DataFrame([inputData.values()], columns=inputData.keys())
    df["Rf"] = df["Followers"].iloc[0] / \
        (df["Followers"].iloc[0] + df["Friends"].iloc[0])
    df["User_mention"] /= df["total_tweets"]
    df["Url"] /= df["total_tweets"]
    df.drop("total_tweets", inplace=True, axis=1)
    df=df.sort_index(axis=1)
    return df.iloc[0]


def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    # Default to 0 if not provided
    print(request.form)

    Friends = int(request.form.get("Friends"))
    Followers = int(request.form.get("Followers"))
    total_tweets = int(request.form.get("total_tweets"))
    User_mention = int(request.form.get("User_mention"))
    Url = int(request.form.get("Url"))

    input_data = preprocess_input(
        Friends, Followers, Url, total_tweets, User_mention)

    prediction = make_prediction([input_data])[0]
    print(prediction)
    return "Human" if prediction else "Bot"


if __name__ == '__main__':
    app.run(debug=True)
