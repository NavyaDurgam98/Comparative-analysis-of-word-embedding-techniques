!pip install fasttext

# Basic packages
import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from pathlib import Path

# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers
import nltk
nltk.download('stopwords')

# Constants
WORDS_COUNT = 10000  # Parameter indicating the number of words in the dictionary
MAX_SEQUENCE_LENGTH = 24  # Maximum number of words in a sequence
BATCH_SIZE = 512  # Batch size for training
NB_START_EPOCHS = 10  # Number of epochs for training

# Function to remove stopwords and mentions from text
def clean_text(input_text):
    stop_words = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stop_words or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)

def remove_mentions(input_text):
    return re.sub(r'@\w+', '', input_text)

# Function to remove hashtags from text
def remove_hashtags(input_text):
    return re.sub(r'#\w+', '', input_text)

# Function to remove URLs from text
def remove_urls(input_text):
    return re.sub(r'http\S+', '', input_text)

# Function for spelling correction (custom mapping)
def spelling_correction(input_text):
    correction_map = {
        'u': 'you',
        'ur': 'your',
        'r': 'are',
        'd': 'the',
        'b': 'be',
        'wl': 'well'
    }
    words = input_text.split()
    corrected_words = [correction_map[word] if word in correction_map else word for word in words]
    return " ".join(corrected_words)

# Load data and preprocess
tweets_df = pd.read_csv('Tweets.csv', encoding='latin1')
tweets_df = tweets_df.reindex(np.random.permutation(tweets_df.index))
columns = ['target', 'id', 'date', 'flag', 'user', 'tweet_text']
tweets_df.columns = columns
tweets_df = tweets_df.drop_duplicates(subset=['tweet_text'])  # Remove duplicates
tweets_df['target'] = tweets_df['target'].replace({2: 'neutral', 0: 'negative', 4: 'positive'})
tweets_df = tweets_df[['tweet_text', 'target']]
tweets_df.tweet_text = tweets_df.tweet_text.apply(clean_text).apply(remove_mentions).apply(remove_hashtags).apply(remove_urls).apply(spelling_correction)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(tweets_df.tweet_text, tweets_df.target, test_size=0.2, random_state=37)

print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])

# Tokenization
tokenizer = Tokenizer(num_words=WORDS_COUNT,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)

# Define embedding dimension
embedding_dim = 300

# Recalculate word count
word_count = min(WORDS_COUNT, len(tokenizer.word_index))

# Create embedding matrix
embedding_matrix = np.zeros((word_count, embedding_dim))

# Load FastText word embeddings
import gensim.downloader as api
word_vectors = api.load("fasttext-wiki-news-subwords-300")

# Fill embedding matrix with FastText embeddings
for word, i in tokenizer.word_index.items():
    if i >= WORDS_COUNT:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)

# Model architecture
model = models.Sequential()
model.add(layers.Embedding(WORDS_COUNT, embedding_dim, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train_cat, epochs=NB_START_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test_cat, verbose=0)
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))

# Predictions
y_pred_probabilities = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Convert one-hot encoded labels back to categorical labels
y_test_cat_labels = np.argmax(y_test_cat, axis=1)

# Calculate precision, recall, and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test_cat_labels, y_pred, average='weighted')
recall = recall_score(y_test_cat_labels, y_pred, average='weighted')
f1 = f1_score(y_test_cat_labels, y_pred, average='weighted')

print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
