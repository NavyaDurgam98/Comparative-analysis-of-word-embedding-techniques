import re
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    tokens = [word for word in word_tokenize(text) if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

df['text'] = df['text'].apply(preprocess_text)

# Tag documents
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df['text'])]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=40)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Vectorize tweets
def vectorize_tweet(tweet, model):
    return model.infer_vector(tweet)

tweet_vecs = np.array([vectorize_tweet(tweet, doc2vec_model) for tweet in df['text']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweet_vecs, df['sentiment'][:len(tweet_vecs)], test_size=0.2, random_state=42)

# Classification with Logistic Regression
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")