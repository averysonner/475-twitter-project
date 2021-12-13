
import math
from datetime import datetime
from os import path
from numpy import minimum

from gensim.models import Doc2Vec
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

import mysql.connector
import query_constants

label_resolution = 25

tweetdb = mysql.connector.connect(
    host="localhost",
    user="access",
    password="password",
    database="twitter_data"
)
dbcursor = tweetdb.cursor()

render_plot = False

def plot_popularity(pop_list):

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.ylabel("Popularity", fontsize = 14)
    plt.title("Tweet Popularity Score (Raw)", fontsize = 18)

    sorted_pop = list(pop_list)
    sorted_pop.sort()
    plt.scatter(list(range(len(sorted_pop))), sorted_pop, s = 1, c = 'blue')  

    plt.savefig("./img/tweets_raw.png")

    plt.clf()

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.ylabel("Popularity", fontsize = 14)
    plt.title("Tweet Popularity Score (Adjusted)", fontsize = 18)

    adjusted_pop = list(map(lambda x: (math.log(x, 10)), sorted_pop))
    
    plt.scatter(list(range(len(adjusted_pop))), adjusted_pop, s = 1, c = 'blue')  

    plt.savefig("./img/tweets_adjusted.png")

    quit()

twitter_epoc_date = datetime(day=21, month=3, year=2006) #tweet id 1
current_day       = datetime.today()

def daysfromzero(date):
    return  (date - twitter_epoc_date).days #days between date and day zero

def format_date(date):
    day_range = daysfromzero(current_day)
    return (day_range-daysfromzero(date))/float(day_range)

def calc_range_label(range, x):
    return math.floor(x/range) + 1


def get_data():
    
    pop_list     = []
    feature_set  = []
    
    d2v_model    = Doc2Vec.load(query_constants.model_file_name)
    dbcursor.execute(query_constants.get_random_rows.format('25000'))
    tweets       = dbcursor.fetchall()

    for row in tweets:
        # row = tweet_id, tweet_text, tweet_likes, tweet_retweets, tweet_replies, created_at
        datapoint = []
        
        likes = int(row[2])
        retwe = int(row[3])
        repli = int(row[4])
        
        pop_list.append((1.5*retwe + likes + .5*repli)+1)

        datapoint.append(format_date(row[5]))

        words = word_tokenize(row[1])

        feature_vec = d2v_model.infer_vector(words)
        
        for feat in feature_vec: datapoint.append(feat)

        feature_set.append(datapoint)
    
    if render_plot:
        plot_popularity(pop_list)


    #label popularity on a exponential-adjusted scale of 1-resolution
    popularity = list(map(lambda x: math.log(x, 10) if x > 1 else x, pop_list))
    
    lowest_pop = min(popularity)
    highest_pop = max(popularity)
    label_range = (highest_pop-lowest_pop)/label_resolution

    popularity_lables = list(map(
            lambda x: calc_range_label(label_range, (x-lowest_pop)), popularity
        ))

    return feature_set, popularity_lables
    
