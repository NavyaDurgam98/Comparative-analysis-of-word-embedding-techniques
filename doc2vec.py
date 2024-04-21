import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models, layers
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Constants for preprocessing and model configuration
Number_of_Words = 10000  # Number of most frequent words to consider
Size_of_Validation = 4000  # Size of validation set
Maximum_Length = 24  # Maximum length of each sequence
Number_of_dimensions = 100  # Dimensionality of the word vectors
Size_of_Batch = 512  # Batch size for training
Number_of_Epochs = 10  # Number of epochs for training


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


# Function to build and train the deep learning model
def deep_model(model, X_train, y_train, X_valid, y_valid):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    metrics = model.fit(X_train, y_train, epochs=Number_of_Epochs, batch_size=Size_of_Batch,
                        validation_data=(X_valid, y_valid), verbose=1)
    return metrics


# Function to test the trained model
def test_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    return results


# Load and preprocess the dataset
df = pd.read_csv('Tweets.csv', encoding='latin1')
df = df.reindex(np.random.permutation(df.index))
column_names = ['target', 'id', 'date', 'flag', 'user', 'tweet_text']
df.columns = column_names
df['target'] = df['target'].replace({2: 'neutral', 0: 'negative', 4: 'positive'})
df = df[['tweet_text', 'target']]
df.tweet_text = df.tweet_text.apply(remove_stopwords).apply(remove_mentions).apply(remove_hashtags).apply(remove_urls).apply(spelling_correction)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.tweet_text, df.target, test_size=0.2, random_state=37)
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print('# Train data samples:', X_train.shape[0])
print('# Test data samples:', X_test.shape[0])

# Tokenize text data
def tokenize_text(text):
    return word_tokenize(text.lower())

# Function to tag documents for Doc2Vec
def tag_docs(docs):
    tagged = []
    for i, doc in enumerate(docs):
        tagged.append(TaggedDocument(words=tokenize_text(doc), tags=[i]))
    return tagged

# Train Doc2Vec model
def train_doc2vec(tagged_docs, vector_size=100, epochs=20):
    model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Function to vectorize documents using Doc2Vec
def vectorize_docs(model, tagged_docs):
    vectors = [model.infer_vector(doc.words) for doc in tagged_docs]
    return vectors

# Tag train and test documents for Doc2Vec
tagged_train_docs = tag_docs(X_train)
tagged_test_docs = tag_docs(X_test)

# Train Doc2Vec model
doc2vec_model = train_doc2vec(tagged_train_docs)

# Vectorize train and test documents using Doc2Vec
X_train_doc2vec = vectorize_docs(doc2vec_model, tagged_train_docs)
X_test_doc2vec = vectorize_docs(doc2vec_model, tagged_test_docs)

# Label encode target variable
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
num_classes = len(le.classes_)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)

# Define the neural network model for Doc2Vec embeddings
doc2vec_nn_model = models.Sequential()
doc2vec_nn_model.add(layers.Dense(64, activation='relu', input_shape=(100,)))
doc2vec_nn_model.add(layers.Dense(64, activation='relu'))
doc2vec_nn_model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
doc2vec_nn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
doc2vec_history = deep_model(doc2vec_nn_model, np.array(X_train_doc2vec), y_train_oh, np.array(X_test_doc2vec), y_test_oh)

# Evaluate the model on test data
test_loss, test_acc = test_model(doc2vec_nn_model, np.array(X_test_doc2vec), y_test_oh)
print('Test accuracy:', test_acc)

# Make predictions and generate classification report
y_pred = np.argmax(doc2vec_nn_model.predict(np.array(X_test_doc2vec)), axis=-1)

# Generate classification report
report = classification_report(y_test_le, y_pred, target_names=le.classes_, output_dict=True)

# Extract metrics from classification report
accuracy = report['accuracy']
precision = report['macro avg']['precision']
recall = report['macro avg']['recall']
f1_score = report['macro avg']['f1-score']

# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)
