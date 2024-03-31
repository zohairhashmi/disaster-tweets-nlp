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

def remove_attherate_word(words):
    return [word for word in words if not word.startswith('@')]

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

def remove_http_link(words):
    url_patterns = r'\b\w*(http|https|www|\.com)\w*\b'
    # matches = re.findall(url_patterns, ' '.join(words))
    text = re.sub(url_patterns, '', ' '.join(words))
    return text.split()

def lemmatisation(words):
    doc = nlp(' '.join(words))
    return [token.lemma_ for token in doc]

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
                else:
                    continue
        else:
            continue
    return text


# Read the data from the csv file
data = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')

print("Number of missing values in keyword column: {} \n".format(data['keyword'].isnull().sum()))
print("Number of missing values in location column: {} \n".format(data['location'].isnull().sum()))
# total records
print("Total records: {} \n".format(len(data)))

# Clean the text column
for t in data['text']:
    words = t.split()
    words = lowercase(words)
    words = remove_attherate_word(words)
    words = remove_http_link(words)
    words = remove_non_alphanumeric(words)
    words = remove_stop_words(words)
    words = lemmatisation(words)
    data.loc[data['text'] == t, 'text'] = ' '.join(words)

# Clean the keyword column
for t in data['keyword']:
    if type(t) == str and '%20' in t:
        data.loc[data['keyword'] == t, 'keyword'] = keyword_replace(t, '%20', ' ')
    words = lowercase(words)
    words = remove_non_alphanumeric(words)
    data.loc[data['keyword'] == t, 'keyword'] = ' '.join(words)
data['keyword'].fillna('', inplace=True)

# Update missing locations
missing_location = data[data['location'].isnull()]
for t in missing_location['text']:
    entity = update_location(t)
    entity = lowercase([entity])[0] if entity else ''
    data.loc[data['text'] == t, 'location'] = entity

# Clean the location column
data['location'].fillna('', inplace=True)
for t in data['location']:
    words = t.split()
    words = lowercase(words)
    words = remove_non_alphabetic(words)
    words = remove_stop_words(words)
    words = remove_http_link(words)
    words = reset_states(words)
    data.loc[data['location'] == t, 'location'] = ' '.join(words)

print(data.head())

data.to_csv('data/train_cleaned.csv', index=False)

# End the timer
end_time = time.time()

# Print time in minutes
print("Time taken: {} minutes".format((end_time - start_time) / 60))

exit(0)