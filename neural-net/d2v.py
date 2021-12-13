from os.path import exists
from random import shuffle

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from nltk.tokenize import word_tokenize

import mysql.connector
import query_constants

tweetdb = mysql.connector.connect(
    host="localhost",
    user="access",
    password="password",
    database="twitter_data"
)

dbcursor = tweetdb.cursor()

#Train the doc2vec model for use in learn.py
if __name__ == '__main__':
    #run quick check for existing model
    
    if(exists(query_constants.model_file_name)):
        print("Model currently exits, please rename or delete current model before creating a new one.")
        quit()
    
    #Get iterator for text data
    dbcursor.execute(query_constants.get_all_text)
    tweets = dbcursor.fetchall()
    print("Fetched all tweets.")

    #Tag up the tweets
    tagged_tweets = []
    for tweet in tweets:
        tagged_tweets.append(TaggedDocument(words=word_tokenize(tweet[1]), tags=[str(tweet[0])]))
    
    #free up some memory in case the training needs it
    del tweets
    
    #define doc2vec model
    d2v_model = Doc2Vec(min_count=1, window=10, vector_size=250, sample=1e-4, negative=5, workers=8, dm=1)
    
    d2v_model.build_vocab(corpus_iterable=tagged_tweets)
    print("Added all tweets to model's vocab.")

    #train up the model
    for epoch in range(100):
        #print(f'epoch: {epoch}')
        shuffle(tagged_tweets)
        d2v_model.train(
            corpus_iterable = tagged_tweets,
            total_examples  = d2v_model.corpus_count,
            epochs          = 1
        )

    print("Model done training")

    d2v_model.save(query_constants.model_file_name)


# Refrence http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
