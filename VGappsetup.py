
import pandas as pd
import streamlit as st
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
import pickle as pkl
from tqdm import tqdm
import re
from summarizer import Summarizer

# Read data
video_games = pd.read_csv(
    'C:/Users\mabul/OneDrive/Documents/Gio/UMN MABA/Machine Learning/ML-FinalProject/Video-games-condensed-small.csv',
    header=0)



# Define stop words
stopwords = list(STOP_WORDS) + ['game', 'games', 'video', 'videos']


# Define functions
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

# Define summarizer
model = Summarizer()
def summarized_review(data):
    data = data.values[0]
    return model(data, num_sentences=3)


class VGRecs:

    def __init__(self):
        # Define embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def clean_data(self):
        # Aggregate all reviews for each hotel
        aggregate_reviews = video_games.sort_values(['title']).groupby('title', sort=False).description.apply(
            ''.join).reset_index(name='description')

        # Review summary
        aggregate_summary = aggregate_reviews.copy()
        aggregate_summary['summary'] = aggregate_summary[["description"]].apply(summarized_review, axis=1)

        # Retain only alpha numeric characters
        aggregate_reviews['description'] = aggregate_reviews['description'].apply(
            lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

        # Remove the [] in the text
        aggregate_reviews['description'] = aggregate_reviews['description'].str.strip('[]')

        # Change to lowercase
        aggregate_reviews['description'] = aggregate_reviews['description'].apply(lambda x: lower_case(x))

        # Remove stop words
        aggregate_reviews['description'] = aggregate_reviews['description'].apply(
            lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

        #Removing NAs
        #aggregate_reviews.dropna(subset=['description'])

        # Retain the parsed review body in the summary df
        aggregate_summary['description'] = aggregate_reviews['description']

        df_sentences = aggregate_reviews.set_index("description")
        df_sentences = df_sentences["title"].to_dict()
        df_sentences_list = list(df_sentences.keys())



        # Embeddings
        corpus = [str(d) for d in tqdm(df_sentences_list)]
        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

        # Dump to pickle file to use later for prediction
        with open("corpus.pkl", "wb") as file1:
            pkl.dump(corpus, file1)

        with open("corpus_embeddings.pkl", "wb") as file2:
            pkl.dump(corpus_embeddings, file2)

        with open("description.pkl", "wb") as file3:
            pkl.dump(aggregate_reviews, file3)

        with open("description1.pkl", "wb") as file4:
            pkl.dump(aggregate_summary, file4)

        return aggregate_summary, aggregate_reviews, corpus, corpus_embeddings

    def construct_app(self):
        aggregate_summary, aggregate_reviews, corpus, corpus_embeddings = self.clean_data()

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
        st.markdown(
            '<p class="header-style"> Hotel Recommender System </p>',
            unsafe_allow_html=True
        )

        # Print summarized text
        st.markdown("Aggregated reviews")
        st.dataframe(aggregate_reviews)
        st.markdown("Aggregated summary")
        st.dataframe(aggregate_summary)
        st.markdown("Corpus")
        st.write(corpus)
        st.markdown("Corpus Embeddings")
        st.write(corpus_embeddings)

        return self


VG = VGRecs()
VG.construct_app()

