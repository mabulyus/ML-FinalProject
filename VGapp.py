#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#websocket.connect(uri, ping_interval=None)

"""


@author: Mohamed Abulyusr@ MABA CLASS
"""

import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
import re
from spacy import displacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title("MABA 6490")
st.header("Final Project: Video Game Search")


st.subheader("♠♣   Mohamed Abulyusr   ♥♦")

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")

st.markdown("## 🎲 The Application")
st.markdown(" ")

st.markdown("##### **♟ User Input ♟**")

st.markdown("* The app will ask the user to enter a genre, buzzword, or kind of game that they wish to find.")

st.markdown(" ")

st.markdown("##### **♟ Modeling ♟**")

st.markdown("* The app will encode the user's input and find a game that has an encoded summary (previously run and saved) that is most similar to the input.")
st.markdown(" ")

st.markdown("##### **♟ Output ♟**")
st.markdown("* The top 3 games that are relevant to the user's search will be shown!")
st.markdown(" ")
st.markdown(" ")


# Define Constants
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


class VGRec:

    #def __init__(self):
        # Define embedder
       # self.embedder = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

    def load_data(self):
        # load the data from pickle files
        corpus = pd.read_pickle('corpus.pkl')
        corpus_embeddings = pd.read_pickle('corpus_embeddings.pkl')
        aggregate_reviews = pd.read_pickle('description.pkl')
        aggregate_summary = pd.read_pickle('description1.pkl')

        return corpus, corpus_embeddings, aggregate_reviews, aggregate_summary

    def construct_sidebar(self):
        # Construct the input sidebar for user to choose the input
        st.sidebar.image('sidebar.png')
        st.sidebar.markdown(
            '<p class="font-style"><b>Video Game Search</b></p>',
            unsafe_allow_html=True
        )

        query = st.sidebar.text_area("What kind of game are you looking for?", 'Zombies')

        if not query:
            return ""

        else:
            return query





    def plot_wordCloud(self, corpus):
        # Create and generate a word cloud image:
        wordcloud = WordCloud(max_font_size=35, max_words=35, background_color="black", colormap='Set2').generate(corpus)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    def get_recs(self):

        # Get hotel recommendations
        q = self.construct_sidebar()
        if not q:
            st.write('You have not searched for anything yet!')
        else:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            query = q
            corpus, corpus_embeddings, aggregate_reviews, aggregate_summary = self.load_data()
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest scores
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=3)

            st.subheader("Here are your top 3 recommendations for your search!")
            st.markdown(" ")
            st.markdown(" ")

            # Find the closest sentences of the corpus for each query sentence based on cosine similarity
            for score, idx in zip(top_results[0], top_results[1]):
                row_dict = aggregate_reviews.loc[aggregate_reviews['description'] == corpus[idx]]['title'].values[0]
                summary = aggregate_summary.loc[aggregate_summary['description'] == corpus[idx]]['summary']
                st.write(HTML_WRAPPER.format(
                    "<b>Game:  </b>" + re.sub(r'[0-9]+', '', row_dict) + "(Score: {:.4f})".format(
                        score) + "<br/><br/><b>Game Summary:  </b>" + summary.values[0]), unsafe_allow_html=True)
                self.plot_wordCloud(corpus[idx])

    def construct_app(self):
        st.image('Best-PS4-Games-2020.jpg')
        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        return self.get_recs()


VG = VGRec()
VG.construct_app()
