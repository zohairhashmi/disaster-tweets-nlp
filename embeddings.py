# Create word and sentence embeddings using BERT and save them to a file
# Usage: python embeddings.py <input_file> <output_file>
# Example: python embeddings.py data.csv embeddings.pkl
import sys
import torch
import pandas as pd
import pickle
from transformers import BertTokenizer, BertModel
import tqdm

# Read input file
input_file = sys.argv[1]  # First argument for input filename
data = pd.read_csv(input_file)
sentences = data['text'].to_list()
embeddings = []

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Track progress
total_sentences = len(sentences)
processed_sentences = 0

# Tokenize the text
for sentence in tqdm.tqdm(sentences):
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
        embeddings.append(last_hidden_states)
    processed_sentences += 1

print(f"Embeddings created for {processed_sentences}/{total_sentences} sentences.")

# Save embeddings to pickle file
output_file = sys.argv[2]  # Second argument for output filename
with open(output_file, 'wb') as f:
    pickle.dump(embeddings, f)

exit(0)