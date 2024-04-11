import sys
import pandas as pd
import numpy as np
import re
import time

import us_state_abbrev as abr
states = abr.us_state_to_abbrev
states = {k.lower(): v.lower() for k, v in states.items()}

import spacy
nlp = spacy.load('en_core_web_sm')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) # Create a list of stop words

# Start the timer
start_time = time.time()

def remove_attherate_word(words): # Filter words that don't contain "@" using list comprehension
    return [word for word in words if '@' not in word]

def remove_stop_words(words): # Remove the stop words from the text
    return [word for word in words if word not in stop_words]

def remove_non_alphanumeric(words): # Remove all the non-alphanumeric characters from the text
    return [re.sub(r'\W+', '', word) for word in words]

def remove_non_alphabetic(words): # Remove all the non-alphabetic characters from the text
    return [re.sub(r'[^a-zA-Z]', '', word) for word in words]

def lowercase(words): # Convert all the words to lowercase
    return [word.lower() for word in words]

def keyword_replace(text, toreplace, replacewith):
    return text.replace(toreplace, replacewith)

def remove_url(words): # Filter words without any URL substring using list comprehension with "any" check
  return [word for word in words if all(x not in word for x in ("http", "www", ".com", "t.co"))]

def lemmatisation(words): # Lemmatize the words
    return [token.lemma_ for token in nlp(' '.join(words))]

def update_location(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == 'GPE'] # select the last entity if exists and is a GPE
    entities = [e for e in entities if e.isalpha()] # remove any non-alphabetic characters
    return entities[-1] if entities else '' # return the last entity if exists and is a GPE

def reset_states(text):
    for i in range(len(text)):
        if text[i] in states.values():
            return text[i]
        elif text[i] in states.keys():
            return states[text[i]]
        elif text[i] == 'new':
            if i + 1 < len(text):
                new_word = text[i] + ' ' + text[i + 1]
                if new_word in states.keys():
                    return states[new_word]
    return text


# Read the data from the csv file
data = pd.read_csv(sys.argv[1])

print("Number of missing values in keyword column: {} \n".format(data['keyword'].isnull().sum()))
print("Number of missing values in location column: {} \n".format(data['location'].isnull().sum()))
print("Total records: {} \n".format(len(data)))

# Clean the text column
print('cleaning text column...')
for i in range(len(data['text'])):
    words = data['text'][i].split()
    words = lowercase(words)
    words = remove_attherate_word(words)
    words = remove_url(words)
    words = remove_non_alphanumeric(words)
    words = remove_stop_words(words)
    words = lemmatisation(words)
    data.loc[i, 'text'] = ' '.join(words)

# Clean the keyword column
print('cleaning keyword column...')
for i in range(len(data['keyword'])):
    if type(data['keyword'][i]) == str and '%20' in data['keyword'][i]:
        data.loc[i, 'keyword'] = keyword_replace(data['keyword'][i], '%20', ' ')
    elif type(data['keyword'][i]) == float:
        data.loc[i, 'keyword'] = ''
    words = lowercase(data['keyword'][i])
    words = remove_non_alphanumeric(words)
    data.loc[i, 'keyword'] = ''.join(words)
data['keyword'].fillna('', inplace=True)

# Update missing locations
print('updating missing locations...')
missing_location = data[data['location'].isnull()]
for t in missing_location['text']:
    entity = update_location(t)
    entity = lowercase([entity])[0] if entity else ''
    data.loc[data['text'] == t, 'location'] = entity

# Clean the location column
print('cleaning location column...')
data['location'].fillna('', inplace=True)
for i in range(len(data['location'])):
    words = data['location'][i].split()
    words = lowercase(words)
    words = remove_non_alphabetic(words)
    words = remove_stop_words(words)
    words = remove_url(words)
    words = reset_states(words)
    data.loc[i, 'location'] = ''.join(words)

print(data.head(10))

data.to_csv(sys.argv[2], index=False)

# End the timer
end_time = time.time()

print("Time taken: {} minutes".format((end_time - start_time) / 60))

exit(0)