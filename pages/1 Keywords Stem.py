import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import spacy
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import StringIO
from nltk.stem.snowball import SnowballStemmer

#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Keywords Stem")
st.subheader('Put your CSV file and choose method')

#===upload===
uploaded_file = st.file_uploader("Choose your a file")
col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
         'Choose method',
       ('Stemming', 'Lemmatization'))
with col2:
    keyword = st.selectbox(
        'Choose column',
       ('Author Keywords', 'Index Keywords'))

#===body===
if uploaded_file is not None:
     papers = pd.read_csv(uploaded_file)
     keywords = papers.dropna(subset=[keyword])
     datakey = keywords[keyword].map(lambda x: re.sub(' ', '_', x))
     datakey = datakey.map(lambda x: re.sub('-—–', '_', x))
     datakey = datakey.map(lambda x: re.sub(';_', ' ', x))
     datakey = datakey.map(lambda x: x.lower())
     
     #===stem/lem===
     if method is 'Lemmatization':          
        lemmatizer = WordNetLemmatizer()
        def lemmatize_words(text):
             words = text.split()
             words = [lemmatizer.lemmatize(word,pos='v') for word in words]
             return ' '.join(words)
        datakey = datakey.apply(lemmatize_words)
             
     else:
        stemmer = SnowballStemmer("english")
        def stem_words(text):
            words = text.split()
            words = [stemmer.stem(word) for word in words]
            return ' '.join(words)
        datakey = datakey.apply(stem_words)
             #st.write(datakey)
     datakey = datakey.map(lambda x: re.sub(' ', '; ', x))
     datakey = datakey.map(lambda x: re.sub('_', ' ', x))
     keywords[keyword] = datakey
     
     st.write(keywords)
     st.write('Congratulations! 🤩 You choose',keyword ,'with',method,'method. Now, you can easily download the result by clicking the button below')
     
     #===download csv===
     def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

     csv = convert_df(keywords)
     st.download_button(
         "Press to Download 👈",
         csv,
         "scopus.csv",
         "text/csv",
         key='download-csv')
