# The following code is executed in the terminal to train the BERT model for text classification
# Usage: python3 bert-classifier.py <path-to-load-train_cleaned.csv> <path-to-save-model.pth>

import os
import sys
import torch
import tqdm
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name) # BERT model - pre-trained model from Hugging Face Transformers
        self.dropout = nn.Dropout(0.2) # dropout layer - randomly zeroes some of the elements of the input tensor with probability p
        # self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 384) # fully connected layer - applies a linear transformation to the incoming data
        self.fc2 = nn.Linear(384, num_classes) # fully connected layer - applies a linear transformation to the incoming data

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # BERT model forward pass - returns a tuple of last hidden states, pooler output, hidden states, and attentions
        pooled_output = outputs.pooler_output # pooled output - output of the model's classifier (hidden state of the first token of the sequence)
        # pooled_output = self.norm(pooled_output) # layer normalization - normalizes the activations of the previous layer for each data point
        x = self.dropout(pooled_output) # dropout layer forward pass - randomly zeroes some of the elements of the input tensor with probability p
        x = nn.ReLU()(self.fc1(x))
        logits = self.fc2(x)
        return logits

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def classify_texts(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return "positive" if preds.item() == 1 else "negative"
    
def main():
    # Set up parameters
    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    max_length = 28
    batch_size = 12
    num_epochs = 10
    learning_rate = 1e-5

    # load data
    df = pd.read_csv(sys.argv[1])
    # df = pd.read_csv('data/train_cleaned.csv')
    texts = df['text'].to_list()
    labels = df['target'].to_list()

    # split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = BERTClassifier(bert_model_name, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, no_deprecation_warning=True, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in tqdm.trange(num_epochs, desc="Training"):
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Epoch: {epoch}, Accuracy: {accuracy}")
        print(report)
        # time.sleep(20)

    torch.save(model.state_dict(), sys.argv[2])

if __name__ == "__main__":
    main()

# exit code 0
exit(0)