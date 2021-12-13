#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tweepy

import mysql.connector

import datetime

import re

#REF https://fairyonice.github.io/extract-someones-tweet-using-tweepy.html

# credentials from https://apps.twitter.com/


if True: #enable db connection
    
    tweetdb = mysql.connector.connect(
        host="localhost",
        user="access",
        password="password",
        database="twitter_data"
    )

    dbcursor = tweetdb.cursor()


twitter_keys = {
    'bearer_token':   'TOKEN',
    'consumer_key':   'KEY',
    'consumer_secret':'SECRET',

}

https_pattern = r'^https?:\/\/.*[\r\n]*'
clean_pattern = re.compile('\W+')


def getuserlist():
    userfile = 'users.txt'
    file     = open(userfile, 'r')
    users    = file.readlines()

    for i in range(len(users)):
        users[i] = users[i].strip("\n ")

    return users


if __name__ == '__main__':

    #Setup access to API
    client = tweepy.Client(bearer_token       = twitter_keys['bearer_token'],
                           consumer_key       = twitter_keys['consumer_key'],
                           consumer_secret    = twitter_keys['consumer_secret'],
                           wait_on_rate_limit = True)
    
    for username in getuserlist():
        oldest_id = None
        userid = client.get_user(username=username, user_auth=False).data['id']
        tweetsLeft = 10000
        while (tweetsLeft > 0):
            total_tweets = 0
            tweets = client.get_users_tweets(
                                        id          = userid,
                                        user_auth   = False,
                                        max_results = 10, 
                                        exclude     = 'retweets',
                                        tweet_fields= 'id,text,created_at,public_metrics',
                                        until_id    = oldest_id
                                       )
            
            if isinstance(tweets.data, type(None)):
                break
            else:
                for tweet in tweets.data:
                    #remove unicode and hyperlinks
                    clean_tweet_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet.text, flags=re.MULTILINE)
                    clean_tweet_text = clean_pattern.sub(' ', clean_tweet_text).strip().lower()
                    
                    try:
                        if len(clean_tweet_text) > 0:
                            #Immediately commit to db
                            #
                            sql_statement = \
                            f"""
                                INSERT INTO tweets (
                                    tweet_id, tweet_text,
                                    tweet_likes, tweet_retweets, tweet_replies,
                                    screen_name, author_id, 
                                    created_at, inserted_at
                                ) VALUES (
                                    {tweet.id},
                                    '{clean_tweet_text}', 
                                    {tweet['public_metrics']['like_count']},
                                    {tweet['public_metrics']['retweet_count']},
                                    {tweet['public_metrics']['reply_count']},
                                    '{username}',
                                    {userid},
                                    '{tweet.created_at.strftime('%Y-%m-%d %H:%M:%S')}',
                                    '{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
                                )
                            """
                            dbcursor.execute(sql_statement)
                            tweetdb.commit()
                            total_tweets = total_tweets + 1
                        
                    except:
                        print(f'Could not insert Tweet {tweet.id}, could be duplicate.')
                #Log info saved to DB
                #Need a proper logging format if so. 
                #Otherwise could pull inserted at times.
                oldest_id = tweets.meta['oldest_id']
                tweetsLeft = tweetsLeft - total_tweets
                print(f"{username}: Obtained {total_tweets} tweets.")
