import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch

# Assuming CUDA is available, if not replace 'cuda' with 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv('Tweets.csv')
# Simplify for binary classification: Positive and Negative. Adapt as necessary.
df = df[df['sentiment'].isin(['positive', 'negative'])]

# Preprocess and tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TweetDataset(Dataset):
    def _init_(self, tweets, labels, tokenizer):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
    
    def _len_(self):
        return len(self.tweets)
    
    def _getitem_(self, idx):
        tweet = self.tweets[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(tweet, return_tensors='pt', padding=True, truncation=True, max_length=256)
        return {**encoding, 'labels': torch.tensor(labels)}

# Encode labels
df['labels'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the data
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['labels'], test_size=0.2)

# Create dataset
train_dataset = TweetDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
test_dataset = TweetDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)

# Load BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluation
def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds)
            real_values.extend(batch['labels'])
    
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
predictions, real_values = get_predictions(model, test_loader)

print('Test Accuracy:', accuracy_score(real_values, predictions))