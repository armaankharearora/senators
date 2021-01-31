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
from nltk.corpus import stopwords
import spacy

import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys

with open('stopwords.txt', 'r', encoding='utf8', errors='ignore') as txtfile:
        stopwords_txt = txtfile.read()
        mystopwords = stopwords_txt.split()

tfidf_stop_words = text.ENGLISH_STOP_WORDS.union(mystopwords)
tfidf_stop_words = [x.lower() for x in tfidf_stop_words]

nlp = spacy.load("en_core_web_sm")

f = open('../../creds.json', 'r')
creds = json.loads(f.read())
auth = tweepy.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
auth.set_access_token(creds["access_token"], creds["access_token_secret"])

api = tweepy.API(auth)

with open('twitterhandles.csv') as csvfile:
     reader = csv.DictReader(csvfile)
     twitterhandles = list()
     for row in reader:
         twitterhandles.append({'handle': row['Twitter Handle'], 'party': row['Party']})

print(twitterhandles)


#preprocess text in tweets by removing links, @UserNames, blank spaces, etc.
def preprocessing_text(tweet):
    #put everythin in lowercase
    #table['tweet'] = table['tweet'].str.lower()
    #Replace rt indicating that was a retweet
    #table['tweet'] = table['tweet'].str.replace('rt', '')
    #Replace occurences of mentioning @UserNames
    #table['tweet'] = table['tweet'].replace(r'@\w+', '', regex=True)
    #Replace links contained in the tweet
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www.[^ ]+', '', tweet)
    #remove numbers
    tweet = re.sub(r'[0-9]+', '', tweet)
    #replace special characters and puntuation marks
    tweet = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', tweet)
    return tweet

docs = []
def refresh_tweets():
    for item in twitterhandles:
        text = ""
        print('********************************Getting Tweets for %s**********************************' % item['handle'])
        public_tweets = api.user_timeline(item['handle'])
        for tweet in public_tweets:
            text +=  preprocessing_text(tweet.text) + "\n"
        print(text)

        f = open(f"raw_data/{item['handle']}.txt", "w+", encoding= "utf-8")
        f.write(text)
        docs.append(text)
        f.close()


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

def refresh_sim_data():
    cv = TfidfVectorizer(stop_words=tfidf_stop_words, ngram_range=(1,2))
    X = np.array(cv.fit_transform(docs).todense())

    euclidean_distance(X[0], X[1])
    cosine_similarity(X[0], X[1])

    i = 0
    for item1 in twitterhandles:
        f = open(f"raw_data/{item1['handle']}.txt", "r", encoding= "utf-8")
        text1 = f.read()
        f.close()

        f = open(f"similarity_data/{item1['handle']}.txt", "w+", encoding= "utf-8", newline='')
        sim_csv = csv.writer(f)
        sim_csv.writerow(['name', 'spacy_sim', 'cosine_sim', 'euclidean_dist', 'party'])
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

def closest_point(a, b, c, x_0, y_0):
    x = (b*(b*x_0 - a*y_0) - a*c) / (a*a + b*b)
    y = (a*(-b*x_0 + a*y_0)-b*c) / (a*a + b*b)
    return (x,y)

def refresh_bipar_indx():
    f = open(f"bipar_scores.csv", "w+", encoding= "utf-8", newline='')
    bipar = csv.writer(f)
    bipar.writerow(['name', 'score', 'party sim', 'other party sim', 'party'])
    for item in twitterhandles:
        party = item['party']
        print(item)
        df =  pd.read_csv(f"similarity_data/{item['handle']}.txt")
        sorted_df = df.sort_values(by=['cosine_sim'], ascending = False)
        party_df = sorted_df[sorted_df['party'] == party]
        not_party_df = sorted_df[sorted_df['party'] != party]
        #st.dataframe(party_df)

        ave_par_sim = party_df['cosine_sim'][:5].mean()
        ave_not_par_sim = not_party_df['cosine_sim'][:5].mean()
        st.subheader('Bipartisan Bridge Index')
        #score = (ave_not_par_sim / ave_par_sim)*100
        cords = closest_point(1, -1, 0, ave_par_sim, ave_not_par_sim)
        score = ((cords[0]+cords[1])/2) * 100
        score_txt = f'This Senators Score is {score:.2f}'
        print(score_txt)
        bipar.writerow([item['handle'], score, ave_par_sim, ave_not_par_sim, party])

if __name__ == "__main__":
    print(sys.argv)
    if 't' in sys.argv:
        refresh_tweets()
    if 's' in sys.argv:
        refresh_sim_data()
    if 'b' in sys.argv:
        refresh_bipar_indx()
