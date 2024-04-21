import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Importing libraries for NLP and Machine Learning
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import nltk
from sklearn.metrics import classification_report
from keras import layers, models

# Define constants and hyperparameters
Number_of_Words = 10000  # Number of words to consider in the dataset
Size_of_Validation = 4000  # Size of validation set
Maximum_Length = 24  # Maximum length of each sequence
Number_of_dimensions = 100  # Number of dimensions for word embeddings
Size_of_Batch = 512  # Batch size for training
Number_of_Epochs = 10  # Number of epochs for training

# Download NLTK stopwords
nltk.download('stopwords')

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


# Function to train the model
def model_training(model, X_train, Y_train, X_valid, Y_valid):
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

# Read the dataset and preprocess it
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

# Tokenize the text
tk = Tokenizer(num_words=Number_of_Words,
               filters='!"#$%&:;<=>?@[\\]^_`{|}~()*+,-./\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(X_train)

# Convert text sequences to sequences of integers
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
X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1, random_state=37)

# Word2Vec model training
model_Word2Vec = Word2Vec(sentences=X_train_to_sequences, vector_size=Number_of_dimensions, window=5, min_count=1)

# Define neural network architecture
model = models.Sequential()
model.add(layers.Embedding(input_dim=Number_of_Words, output_dim=Number_of_dimensions, input_length=Maximum_Length))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Print model summary
model.summary()

# Train the model
history = model_training(model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_seq_trunc, y_test_oh)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions on test data
y_pred = model.predict(X_test_seq_trunc)

# Convert predicted labels and true labels to one-hot encoding
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test_oh, axis=1)

# Generate classification report
classification_rep = classification_report(y_test_labels, y_pred_labels, output_dict=True)

precision = classification_rep['weighted avg']['precision']
recall = classification_rep['weighted avg']['recall']
f1_score = classification_rep['weighted avg']['f1-score']

# Print evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
