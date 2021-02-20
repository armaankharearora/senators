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


import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS
import re

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim
import sys

from sklearn.cluster import KMeans
import json

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
        f.close()

def read_tweet_docs():
    docs = []
    for item in twitterhandles:
        f = open(f"raw_data/{item['handle']}.txt", "r", encoding= "utf-8")
        text = f.read()
        docs.append(text)
        f.close()
    return docs


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

def tweet_clusters():
    docs = read_tweet_docs()
    cv = TfidfVectorizer(stop_words=tfidf_stop_words, ngram_range=(1,2), min_df = 0.1, max_df = 0.8)
    tfidf_matrix = cv.fit_transform(docs)
    terms = cv.get_feature_names()

    dist = 1 - sk_cos_sim(tfidf_matrix)

    num_clusters = 10
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    print(clusters)
    senators = {'name': twitterhandles, 'cluster': clusters}
    frame = pd.DataFrame(senators, index = [clusters] , columns = ['name', 'cluster'])
    print(frame.head())


    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    cluster_array = []
    for i in range(num_clusters):
        cluster = {"words": [], "senators": []}
        #print("Cluster %d words:" % i, end='')

        for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
            #print(' %s' % terms[ind], end=',')
            cluster["words"].append(terms[ind])
        print() #add whitespace
        print() #add whitespace

        #print("Cluster %d names:" % i, end='')
        for title in frame.loc[i]['name'].values.tolist():
            #print(' %s,' % title, end='')
            cluster["senators"].append(title)

        cluster_array.append(cluster)
        #print(json.dumps(cluster))
        print() #add whitespace
        print() #add whitespace

    print()
    print()
    print(json.dumps(cluster_array))
    cluster_file = open("cluster_file.json", "w")
    cluster_file.write(json.dumps(cluster_array))
    cluster_file.close()

    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    #set up cluster names using a dict
    cluster_names = {0: 'Family, home, war',
                     1: 'Police, killed, murders',
                     2: 'Father, New York, brothers',
                     3: 'Dance, singing, love',
                     4: 'Killed, soldiers, captain'}

    MDS()

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]
    print()
    print()
        #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=twitterhandles))

    #group by cluster
    groups = df.groupby('label')


    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        #ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
        #        label=cluster_names[name], color=cluster_colors[name],
        #        mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)



    #plt.show() #show the plot

    #uncomment the below to save the plot if need be
    #plt.savefig('clusters_small_noaxes.png', dpi=200)

def refresh_sim_data():
    docs = read_tweet_docs()
    cv = TfidfVectorizer(stop_words=tfidf_stop_words, ngram_range=(1,2), min_df = 0.05)
    tfidf_matrix = cv.fit_transform(docs)
    X = np.array(tfidf_matrix.todense())

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
    if "c" in sys.argv:
        tweet_clusters()
