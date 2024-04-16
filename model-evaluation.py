# The following code is executed in the terminal to evaluate the BERT model for text classification
# Usage: python3 model-evaluation.py <path-to-load-model.pth> <path-to-load-test_cleaned.csv> <path-to-save-results.csv>

import sys
import torch
import tqdm
import time
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from BERTClassifier import BERTClassifier # Import the BERTClassifier class from BERTClassifier.py
    
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
