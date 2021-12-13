DROP TABLE IF EXISTS tweets;
CREATE TABLE tweets (
  id INT PRIMARY KEY AUTO_INCREMENT,
  
  tweet_id BIGINT NOT NULL UNIQUE,
  tweet_text VARCHAR(400) NOT NULL,
  
  tweet_likes INT,
  tweet_retweets INT,
  tweet_replies INT,

  screen_name VARCHAR(160) NOT NULL,
  author_id BIGINT,
  created_at DATETIME NOT NULL,
  inserted_at DATETIME NOT NULL
);
