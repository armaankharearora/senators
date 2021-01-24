import streamlit as st
import csv
import tweepy
import json

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import networkx as nx
import pylab as pyl
import streamlit.components.v1 as components

from pyvis.network  import Network
st.set_option('deprecation.showPyplotGlobalUse', False)

add_selectbox = st.sidebar.selectbox(
    "What would you like to see?",
    ('Senator View', 'Summary')
)
if add_selectbox == 'Summary':
    bipar_df = pd.read_csv('model/bipar_scores.csv')
    st.dataframe(bipar_df)

else:
    def closest_point(a, b, c, x_0, y_0):
        x = (b*(b*x_0 - a*y_0) - a*c) / (a*a + b*b)
        y = (a*(-b*x_0 + a*y_0)-b*c) / (a*a + b*b)
        return (x,y)

    with open('model/twitterhandles.csv') as csvfile:
         reader = csv.DictReader(csvfile)
         senator_names = list()
         for row in reader:
             senator_names.append(row['Name'])
    st.title("Senator's Twitter View")

    option = st.selectbox(
         'Senator to analyze?',
         senator_names)
    st.write('You selected:', option)

    with open('model/twitterhandles.csv') as csvfile:
         reader = csv.DictReader(csvfile)
         twitterhandle = None
         party = None
         for row in reader:
             if row['Name'] == option:
                 twitterhandle = row['Twitter Handle']
                 party = row['Party']

    df =  pd.read_csv(f"model/similarity_data/{twitterhandle}.txt")
    sorted_df = df.sort_values(by=['cosine_sim'], ascending = False)

    party_df = sorted_df[sorted_df['party'] == party]
    not_party_df = sorted_df[sorted_df['party'] != party]
    #st.dataframe(party_df)

    ave_par_sim = party_df['cosine_sim'][:5].mean()
    ave_not_par_sim = not_party_df['cosine_sim'][:5].mean()
    st.subheader('Bipartisan Bridge Index')
    #score = (ave_not_par_sim / ave_par_sim)*100
    cords = closest_point(0.75, -1, 0, ave_par_sim, ave_not_par_sim)
    score = ((cords[0]+cords[1])/2) * 100
    score_txt = f'This Senators Score is {score:.2f}'
    st.markdown(score_txt)

    text = ""
    #public_tweets = api.user_timeline(twitterhandle)
    #for tweet in public_tweets:
        #print(tweet.text)
        #text = text + " " + tweet.text
    st.subheader('Wordcloud representing the most used words from the Senators recent tweets')

    f = open(f"model/raw_data/{twitterhandle}.txt", "r", encoding= "utf-8")
    text = f.read()
    f.close()
    stopwords = set(STOPWORDS)
    stopwords.update(["may", "US", "https", "t", "co", "RT", "S", 'U', 'amp', 'must' 'will', 've', 'si02', 'PPP', 'FY21', 'GovKemp', 'el', 'Si', 're', 'QampA', 'the', 'fwd', 'ovr'])
    wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()

    top_5 = df.nlargest(5, 'cosine_sim')
    st.write(top_5)

    G=nx.Graph()

    G.add_node(twitterhandle)

    for index, row in top_5.iterrows():
        if row['name'] != twitterhandle:
            if row['party'] == 'R':
                party_color = 'red'
            else:
                party_color ='blue'
            G.add_node(row['name'], color = party_color)
            G.add_edges_from([(twitterhandle, row['name'], {'color': 'grey','weight': row['cosine_sim']})])

            bof_df = pd.read_csv(f"model/similarity_data/{row['name']}.txt")
            bof_top_5 = bof_df.nlargest(5, 'cosine_sim')
            for bof_index, bof_row in bof_top_5.iterrows():
                if bof_row['name'] != row['name']:
                    if row['party'] == 'R':
                        party_color = 'red'
                    else:
                        party_color ='blue'
                    G.add_node(bof_row['name'], color = party_color)
                    G.add_edges_from([(row['name'], bof_row['name'], {'color': 'grey','weight': bof_row['cosine_sim']})])

    pos = nx.spring_layout(G, weight = 'weight')
    nx.draw_networkx(G, pos)
    pyl.savefig('bof.png')
    #st.image('bof.png')

    nt = Network('750px', '750px', notebook=True, heading = "")
    nt.from_nx(G)
    nt.force_atlas_2based()
    options =  {
       "nodes":{
          "shape":"dot",
          "size":15,
          "font":{
             "size":16
          },
          "borderWidth":2
       }
    }
    nt.options = options
    nt.show('bof.html')
    html_file = open('bof.html', 'r')
    source_code = html_file.read()
    components.html(source_code, height = 800, width = 800)


    st.subheader('Table comparing the similarity of tweets to all the other senators')
    st.markdown("Uses 3 different metrics - Spacy similarity (using word vectors), Cosine similarity, and Euclidean Distance to compare the senator to others.")
    st.dataframe(df)  # Same as st.write(df)
