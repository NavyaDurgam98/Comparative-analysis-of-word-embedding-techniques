import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Importing necessary libraries and modules
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras import models, layers

import nltk
nltk.download('stopwords')

# Define constants and hyperparameters
Number_of_Words = 10000
Size_of_Validation = 4000
Maximum_Length = 24
Number_of_dimensions = 100
Size_of_Batch = 512
Number_of_Epochs = 10

# Function to remove stopwords from text
def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]  # Whitelist of words to keep even if they are stopwords
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)


# Function to remove mentions (@username) from text
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


# Function to train deep learning model
def deep_model(model, X_train, Y_train, X_valid, Y_valid):
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    metrics = model.fit(X_train,
                        Y_train,
                        epochs=Number_of_Epochs,
                        batch_size=Size_of_Batch,
                        validation_data=(X_valid, Y_valid),
                        verbose=1)
    return metrics

# Function to test trained model
def test_model(model, X_train, Y_train, X_test, Y_test, epoch_number):
    model.fit(X_train,
              Y_train,
              epochs=epoch_number,
              batch_size=Size_of_Batch,
              verbose=0)
    results = model.evaluate(X_test, Y_test)

    return results

# Read dataset and preprocess text
df = pd.read_csv('Tweets.csv', encoding='latin1')
column_names = ['target', 'id', 'date', 'flag', 'user', 'tweet_text']
df.columns = column_names
df = df.drop_duplicates(subset=['tweet_text'])  # Remove duplicates
df = df.reindex(np.random.permutation(df.index))
df['target'] = df['target'].replace({2: 'neutral', 0: 'negative', 4: 'positive'})
df = df[['tweet_text', 'target']]
df.tweet_text = df.tweet_text.apply(remove_stopwords).apply(remove_mentions).apply(remove_hashtags).apply(remove_urls).apply(spelling_correction)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df.tweet_text, df.target, test_size=0.2, random_state=37)
assert X_train.shape[0] == Y_train.shape[0]
assert X_test.shape[0] == Y_test.shape[0]
print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])

# Tokenize text
tk = Tokenizer(num_words=Number_of_Words,
               filters='!"#$%&():;<=>?@[\\]^_`{|}*+,-./~\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(X_train)

X_train_to_sequences = tk.texts_to_sequences(X_train)
X_test_to_sequences = tk.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_seq_trunc = pad_sequences(X_train_to_sequences, maxlen=Maximum_Length)
X_test_seq_trunc = pad_sequences(X_test_to_sequences, maxlen=Maximum_Length)

# Encode target labels
le = LabelEncoder()
y_train_le = le.fit_transform(Y_train)
y_test_le = le.transform(Y_test)
num_classes = len(le.classes_)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)

# Split training data into training and validation sets
X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.2, random_state=37)
assert X_valid_emb.shape[0] == y_valid_emb.shape[0]
assert X_train_emb.shape[0] == y_train_emb.shape[0]

# Download GloVe word embeddings
!wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
!unzip glove.twitter.27B.zip

# Load GloVe embeddings
glove_file = 'glove.twitter.27B.' + str(Number_of_dimensions) + 'd.txt'
emb_dict = {}
glove = open(glove_file)
for line in glove:
    value = line.split()
    word = value[0]
    vec = np.asarray(value[1:], dtype='float32')
    emb_dict[word] = vec
glove.close()

# Create embedding matrix using GloVe embeddings
emb_matrix = np.zeros((Number_of_Words, Number_of_dimensions))
for w, i in tk.word_index.items():
    if i < Number_of_Words:
        vect = emb_dict.get(w)
        if vect is not None:
            emb_matrix[i] = vect
    else:
        break

# Build and compile model with GloVe embeddings
glove_model = models.Sequential()
glove_model.add(layers.Embedding(Number_of_Words, Number_of_dimensions, input_length=Maximum_Length))
glove_model.add(layers.Flatten())
glove_model.add(layers.Dense(num_classes, activation='softmax'))
glove_model.summary()

# Set GloVe embedding weights and freeze embedding layer
glove_model.layers[0].set_weights([emb_matrix])
glove_model.layers[0].trainable = False

# Train GloVe model
glove_history = deep_model(glove_model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)
glove_history.history['accuracy'][-1]

# Make predictions and evaluate GloVe model
y_pred = glove_model.predict(X_test_seq_trunc)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test_oh, axis=1)

classification_rep = classification_report(y_test_labels, y_pred_labels, output_dict=True)
precision = classification_rep['weighted avg']['precision']
recall = classification_rep['weighted avg']['recall']
f1_score = classification_rep['weighted avg']['f1-score']

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

# Test accuracy of GloVe model with a smaller number of epochs
glove_results = test_model(glove_model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 3)
print('Test accuracy of word glove model: {0:.2f}%'.format(glove_results[1]*100))
