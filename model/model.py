#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import csv
import tweepy
import json

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import spacy

import matplotlib.pyplot as plt


f = open('../../creds.json', 'r')
creds = json.loads(f.read())
auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
auth.set_access_token(creds["access_token"], creds["access_token_secret"])

api = tweepy.API(auth)


# In[3]:


with open('twitterhandles.csv') as csvfile:
     reader = csv.DictReader(csvfile)
     twitterhandles = list()
     for row in reader:
         twitterhandles.append({'handle': row['Twitter Handle'], 'party': row['Party']})

print(twitterhandles)


# In[7]:


docs = []
for item in twitterhandles:
    text = ""
    print('********************************Getting Tweets for %s**********************************' % item['handle'])
    public_tweets = api.user_timeline(item['handle'])
    for tweet in public_tweets:
        text += tweet.text + "\n"
    print(text)

    f = open(f"raw_data/{item['handle']}.txt", "w+", encoding= "utf-8")
    f.write(text)
    docs.append(text)
    f.close()


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def process_text(text):
    doc = nlp(text.lower())
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(token.lemma_)
    return " ".join(result)

def calculate_similarity(text1, text2):
    base = nlp(process_text(text1))
    compare = nlp(process_text(text2))
    return base.similarity(compare)


# In[9]:


cv = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = np.array(cv.fit_transform(docs).todense())

euclidean_distance(X[0], X[1])
cosine_similarity(X[0], X[1])


# In[11]:


i = 0
for item1 in twitterhandles:
    f = open(f"raw_data/{item1['handle']}.txt", "r", encoding= "utf-8")
    text1 = f.read()
    f.close()

    f = open(f"similarity_data/{item1['handle']}.txt", "w+", encoding= "utf-8", newline='')
    sim_csv = csv.writer(f)
    sim_csv.writerow(['name', 'spacy_sim', 'cosine_sim', 'euclidean_sim', 'party'])
    j = 0
    for item2 in twitterhandles:
        f2 = open(f"raw_data/{item2['handle']}.txt", "r", encoding= "utf-8")
        text2 = f2.read()
        f2.close()
        s_sim = calculate_similarity(text1, text2)
        e_sim = euclidean_distance(X[i], X[j])
        c_sim = cosine_similarity(X[i], X[j])
        print(item1['handle'], item2['handle'], s_sim, c_sim, e_sim, i, j, item2['party'])
        sim_csv.writerow([item2['handle'], s_sim, c_sim, e_sim, item2['party']])
        j += 1

    i += 1



# In[ ]:





# In[ ]:





# In[ ]:
