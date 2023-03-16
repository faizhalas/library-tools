import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from pprint import pprint
import spacy
import pickle
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
     keywords = pd.read_csv(uploaded_file)
     keywords[keyword] = keywords[keyword].astype(str)
     keywords[keyword] = keywords[keyword].map(lambda x: re.sub('-—–', ' ', x))
     keywords[keyword] = keywords[keyword].map(lambda x: x.lower())
     
     #===Keywords list===
     key = keywords[keyword]
     key = key.dropna()
     key = pd.concat([key.str.split('; ', expand=True)], axis=1)
     key = pd.Series(np.ravel(key)).dropna().drop_duplicates().sort_values().reset_index()
     key['new']=key[0]
     
     #===stem/lem===
     if method is 'Lemmatization':          
        lemmatizer = WordNetLemmatizer()
        def lemmatize_words(text):
             words = text.split()
             words = [lemmatizer.lemmatize(word) for word in words]
             return ' '.join(words)
        keywords[keyword] = keywords[keyword].apply(lemmatize_words)
        key['new'] = key['new'].apply(lemmatize_words)
             
     else:
        stemmer = SnowballStemmer("english")
        def stem_words(text):
            words = text.split()
            words = [stemmer.stem(word) for word in words]
            return ' '.join(words)
        keywords[keyword] = keywords[keyword].apply(stem_words)
        key['new'] = key['new'].apply(stem_words)
     
     st.write('Congratulations! 🤩 You choose',keyword ,'with',method,'method. Now, you can easily download the result by clicking the button below')
     
     #===show & download csv===
     col1, col2 = st.columns(2)
     with col1:
         st.write(keywords, use_container_width=True)
         def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

         csv = convert_df(keywords)
         st.download_button(
             "Press to download result 👈",
             csv,
             "scopus.csv",
             "text/csv")
          
     with col2:
         keywords[keyword] = keywords[keyword].map(lambda x: re.sub('nan', '', x))
         key = key.drop(['index'], axis=1).rename(columns={0: 'old'})
         st.write(key, use_container_width=True)
                  
         def convert_dfs(df):
                return df.to_csv(index=False).encode('utf-8')

         csv = convert_dfs(key)
         st.download_button(
             "Press to download keywords 👈",
             csv,
             "keywords.csv",
             "text/csv")
     
     
