import pandas as pd

data = pd.read_csv('data/train_cleaned.csv')
# print all locations having target as 1
data[data['target'] == 1]['location'].unique()
# print all locations having target as 0
# print(data[data['target'] == 0]['location'].value_counts())