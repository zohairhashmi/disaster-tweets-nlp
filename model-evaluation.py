# Test model on test dataset
# Usage: python3 test.py test_cleaned.csv model.pth output.csv
import sys
import torch
import tqdm
import time
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name) # BERT model - pre-trained model from Hugging Face Transformers
        self.dropout = nn.Dropout(0.2) # dropout layer - randomly zeroes some of the elements of the input tensor with probability p
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 384) # fully connected layer - applies a linear transformation to the incoming data
        self.fc2 = nn.Linear(384, num_classes) # fully connected layer - applies a linear transformation to the incoming data

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # BERT model forward pass - returns a tuple of last hidden states, pooler output, hidden states, and attentions
        pooled_output = outputs.pooler_output # pooled output - output of the model's classifier (hidden state of the first token of the sequence)
        x = self.dropout(pooled_output) # dropout layer forward pass - randomly zeroes some of the elements of the input tensor with probability p
        x = nn.ReLU()(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
def classify_texts(text, model, tokenizer, device, max_length=128):
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return preds.item()


df = pd.read_csv(sys.argv[2])
texts = df['text'].to_list()

model_results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
max_length = 30
num_classes = 2

model = BERTClassifier(bert_model_name, num_classes)
model.load_state_dict(torch.load(sys.argv[1]))
model.to(device)
model.eval()

for text in tqdm.tqdm(texts):
    if type(text) != str:
        model_results.append(0)
    else:
        model_results.append(classify_texts(text, model, tokenizer, device, max_length=30))

df['model_results'] = model_results
df.to_csv(sys.argv[3], index=False)
