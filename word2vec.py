import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load data
df = pd.read_csv('Tweets.csv')

# Preprocess data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove punctuation
    text = text.lower()  # Lowercase
    text = text.strip()  # Remove whitespaces
    # Tokenization and removing stopwords
    tokens = [word for word in word_tokenize(text) if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

df['text'] = df['text'].apply(preprocess_text)

# Prepare data for Word2Vec training
tweets = [row.split() for row in df['text']]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tweets, vector_size=100, window=5, min_count=1, workers=4)

# Function to vectorize tweets
def tweet_vectorizer(tweet, model):
    vec = []
    numw = 0
    for w in tweet:
        try:
            if numw == 0:
                vec = model.wv[w]
            else:
                vec = np.add(vec, model.wv[w])
            numw += 1
        except:
            pass
    return np.asarray(vec) / numw

# Vectorize tweets
tweet_vecs = np.array([tweet_vectorizer(tweet, word2vec_model) for tweet in tweets])

# Remove NaN values
tweet_vecs = np.array([vec for vec in tweet_vecs if str(vec) != 'nan'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweet_vecs, df['sentiment'][:len(tweet_vecs)], test_size=0.2, random_state=42)

# Classification with Logistic Regression
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")