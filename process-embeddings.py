##### NOT IMPORTANT FOR THE PROJECT #####

import pandas as pd
import torch
import pickle
import torch.nn.functional as F

with open('data/train_embeddings.pkl','rb') as file:
    embeddings = pickle.load(file)

# Apply mean pooling to convert all embeddings into same standardised (1,768) shape
pooled_embedding = []
for emb in embeddings:
    pooled = torch.mean(embeddings[0], dim=1)  # Shape: (1, 768)
    pooled_embedding.append(pooled)
print('Pooling Completed . . .')

# Normalize the pooled embeddings to achieve a zero mean for all embeddings
normalized_embeddings = []
for pemb in pooled_embedding:
    norm = F.normalize(pemb, p=2, dim=1)
    normalized_embeddings.append(norm)
print('Normalization Completed . . .')

# Save embeddings to pickle file
output_file = 'data/norm_embeddings.pkl'  # Second argument for output filename
with open(output_file, 'wb') as f:
    pickle.dump(normalized_embeddings, f)
print(f'Output File save as {output_file}')

exit(0)
