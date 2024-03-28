import fasttext
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the data
df = pd.read_csv('Tweets.csv')

# Function to clean the tweets
def preprocess_tweet(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['text'] = df['text'].apply(preprocess_tweet)

# Assuming binary sentiment classification with labels 'positive' and 'negative'
# Adjust labels as necessary
df['sentiment'] = df['sentiment'].replace({'positive': '_labelpositive', 'negative': 'label_negative'})

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare data for FastText
train_file = 'fasttext_train.txt'
test_file = 'fasttext_test.txt'
train_df[['sentiment', 'text']].to_csv(train_file, index=False, sep=' ', header=False, quoting=3)
test_df[['sentiment', 'text']].to_csv(test_file, index=False, sep=' ', header=False, quoting=3)

# Train FastText model
model = fasttext.train_supervised(input=train_file, wordNgrams=2, epoch=25, lr=1.0)

# Test the model (get the precision and recall)
def print_results(N, p, r):
    print("Number of examples: ", N)
    print("Precision: {:.2f}".format(p))
    print("Recall: {:.2f}".format(r))

print_results(*model.test(test_file))

# Predicting on new examples
def predict_sentiment(model, text):
    cleaned_text = preprocess_tweet(text)
    prediction = model.predict(cleaned_text)
    return prediction

# Example prediction
example_tweet = "This is a great day!"
print(predict_sentiment(model, example_tweet))