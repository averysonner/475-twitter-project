
model_file_name = './tweets.d2v'

data_request_size = 10000

get_random_rows = \
'''
SELECT 
tweet_id, tweet_text,
tweet_likes, tweet_retweets,
tweet_replies, created_at
FROM tweets
ORDER BY RAND()
LIMIT {}
'''

get_all_text = \
f'''
SELECT author_id, tweet_text FROM tweets;
'''

