import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import style
import tweepy
from textblob import TextBlob 
from models import conf

import csv

style.use('ggplot')

def get_data(filename, dates, prices):
    """
    df = pd.read_csv(filename)
    dates = df.index.tolist()
    prices = df.Open.tolist()
    """
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[2]))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1))

    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def login_to_twitter():
    auth = tweepy.OAuthHandler(conf.CONSUMER_KEY, conf.CONSUMER_SECRET)
    auth.set_access_token(conf.ACCESS_TOKEN, conf.ACCESS_TOKEN_SECRET)
    user = tweepy.API(auth)
    return user

def twitter_sentiment(quote, num_tweets=100, tweets_since='2018-01-01', tweets_until='2018-02-28'):
    """
    Logs into twitter via tweepy, and counts the positivity and objectivity of tweets.
    Returns True if most tweets are positive.
    """
    user = login_to_twitter() 

    tweets = user.search(quote, count=num_tweets, since=tweets_since, until=tweets_until)
    positive_sent, null_sent = 0, 0
    for tweet in tweets:
        print(tweet.user.screen_name)
        print(tweet.created_at)
        print(tweet.text)
        print()
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity is 0:
            null_sent += 1
            next
        if blob.polarity > 0:
            positive_sent += 1

    if positive_sent > ((num_tweets - null_sent)/2):
        print("This stock has positive sentiment based on analyzed tweets")
        return True
    else:
        print("Bad sentiment on stock based on gathered tweets!")
        return False
