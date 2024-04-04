# Create word and sentence embeddings using BERT and save them to a file
# Usage: python3 embeddings.py <input_file> <output_file>

import sys
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()

# Read input file
input_file = sys.argv[1]
data = pd.read_csv(input_file)
sentences = data['text'].to_list()

print(sentences)
# print(data.head())
exit(0)