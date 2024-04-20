# Basic packages
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from transformers import BertTokenizer
import nltk
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
nltk.download('stopwords')

# Constants
WORD_LIMIT = 10000  # Parameter indicating the number of words in the dictionary
MAX_SEQUENCE_LENGTH = 24  # Maximum number of words in a sequence

def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)

def remove_mentions(input_text):
    return re.sub(r'@\w+', '', input_text)

def remove_hashtags(input_text):
    return re.sub(r'#\w+', '', input_text)

def remove_urls(input_text):
    return re.sub(r'http\S+', '', input_text)

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

# Data cleaning and preparation
df = pd.read_csv('Tweets.csv', encoding='latin1')
df = df.reindex(np.random.permutation(df.index))
column_names = ['target', 'id', 'date', 'flag', 'user', 'tweet_text']
df.columns = column_names
df = df.drop_duplicates(subset=['tweet_text'])
df['target'] = df['target'].replace({2: 'neutral', 0: 'negative', 4: 'positive'})
df = df[['tweet_text', 'target']]
df.tweet_text = df.tweet_text.apply(remove_stopwords).apply(remove_mentions).apply(remove_hashtags).apply(remove_urls).apply(spelling_correction)

# Train and Test Split
X_train, X_test, Y_train, Y_test = train_test_split(df.tweet_text, df.target, test_size=0.2, random_state=37)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text data using BERT tokenizer
X_train_bert = [tokenizer.encode(text, add_special_tokens=True, max_length=MAX_SEQUENCE_LENGTH, truncation=True) for text in X_train]
X_test_bert = [tokenizer.encode(text, add_special_tokens=True, max_length=MAX_SEQUENCE_LENGTH, truncation=True) for text in X_test]

# Pad sequences to ensure uniform length
X_train_bert_padded = pad_sequences(X_train_bert, maxlen=MAX_SEQUENCE_LENGTH, dtype="long", value=0, truncating="post", padding="post")
X_test_bert_padded = pad_sequences(X_test_bert, maxlen=MAX_SEQUENCE_LENGTH, dtype="long", value=0, truncating="post", padding="post")

# Converting target labels to numbers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(Y_train)
y_test_encoded = label_encoder.transform(Y_test)
num_classes = len(label_encoder.classes_)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Model Architecture with BERT embeddings and Conv1D
model = models.Sequential()
model.add(layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32"))
model.add(layers.Embedding(len(tokenizer), 768, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

# Compile and train the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_bert_padded, y_train_categorical, epochs=20, batch_size=512, validation_split=0.1, verbose=1)

# Evaluate the model
y_pred_classes = np.argmax(model.predict(X_test_bert_padded), axis=1)
y_test_labels = label_encoder.transform(Y_test)

# Calculate metrics
accuracy = accuracy_score(y_test_labels, y_pred_classes)
precision = precision_score(y_test_labels, y_pred_classes, average='macro')
recall = recall_score(y_test_labels, y_pred_classes, average='macro')
f1 = f1_score(y_test_labels, y_pred_classes, average='macro')

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
