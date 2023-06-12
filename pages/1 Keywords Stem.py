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
import pickle
import streamlit.components.v1 as components
from io import StringIO
from nltk.stem.snowball import SnowballStemmer
import csv
import sys

#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Keywords Stem")
st.subheader('Put your file here...')

def reset_data():
     st.cache_data.clear()

#===upload===
@st.cache_data(ttl=3600)
def upload(file):
    #uploaded_file = file
    keywords = pd.read_csv(file)
    return keywords

@st.cache_data(ttl=3600)
def conv_txt(file):
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DE': 'Author Keywords',
            'ID': 'Keywords Plus'}
    keywords = pd.read_csv(file, sep='\t', lineterminator='\r')
    keywords.rename(columns=col_dict, inplace=True)
    return keywords

@st.cache_data(ttl=3600)
def rev_conv_txt():
    col_dict_rev = {'Title': 'TI',
            'Source title': 'SO',
            'Author Keywords': 'DE',
            'Keywords Plus': 'ID'}
    keywords.rename(columns=col_dict_rev, inplace=True)
    return keywords

@st.cache_data(ttl=3600)
def get_ext(file):
    extype = file.name
    return extype

@st.cache_data(ttl=3600)
def get_data():
    list_of_column_key = list(keywords.columns)
    list_of_column_key = [k for k in list_of_column_key if 'Keyword' in k]
    return list_of_column_key

uploaded_file = st.file_uploader("Choose your a file", type=['csv','txt'], on_change=reset_data)

if uploaded_file is not None:
     extype = get_ext(uploaded_file)
     if extype.endswith('.csv'):
         keywords = upload(uploaded_file) 
                  
     elif extype.endswith('.txt'):
         keywords = conv_txt(uploaded_file)
         
     list_of_column_key = get_data()

     col1, col2 = st.columns(2)
     with col1:
        method = st.selectbox(
             'Choose method',
           ('Stemming', 'Lemmatization'), on_change=reset_data)
     with col2:
        keyword = st.selectbox(
            'Choose column',
           (list_of_column_key), on_change=reset_data)

     @st.cache_data(ttl=3600)
     def clean_keyword():      
        global keyword, keywords
        try:
            key = keywords[keyword]
        except KeyError:
            st.error('Error: Please check your Author/Index Keywords column.')
            sys.exit(1)
        keywords = keywords.replace(np.nan, '', regex=True)
        keywords[keyword] = keywords[keyword].astype(str)
        keywords[keyword] = keywords[keyword].map(lambda x: re.sub('-', ' ', x))
        keywords[keyword] = keywords[keyword].map(lambda x: re.sub('; ', ' ; ', x))
        keywords[keyword] = keywords[keyword].map(lambda x: x.lower())
        
        #===Keywords list===
        key = key.dropna()
        key = pd.concat([key.str.split('; ', expand=True)], axis=1)
        key = pd.Series(np.ravel(key)).dropna().drop_duplicates().sort_values().reset_index()
        key[0] = key[0].map(lambda x: re.sub('-', ' ', x))
        key['new']=key[0].map(lambda x: x.lower())

        return keywords, key
     
     #===stem/lem===
     @st.cache_data(ttl=3600)
     def Lemmatization():
        lemmatizer = WordNetLemmatizer()
        def lemmatize_words(text):
            words = text.split()
            words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(words)
        keywords[keyword] = keywords[keyword].apply(lemmatize_words)
        key['new'] = key['new'].apply(lemmatize_words)
        keywords[keyword] = keywords[keyword].map(lambda x: re.sub(' ; ', '; ', x))
        return keywords, key
                
     @st.cache_data(ttl=3600)
     def Stemming():
        stemmer = SnowballStemmer("english")
        def stem_words(text):
            words = text.split()
            words = [stemmer.stem(word) for word in words]
            return ' '.join(words)
        keywords[keyword] = keywords[keyword].apply(stem_words)
        key['new'] = key['new'].apply(stem_words)
        keywords[keyword] = keywords[keyword].map(lambda x: re.sub(' ; ', '; ', x))
        return keywords, key
     
     keywords, key = clean_keyword() 
     
     if method is 'Lemmatization':
         keywords, key = Lemmatization()
     else:
         keywords, key = Stemming()
            
     st.write('Congratulations! 🤩 You choose',keyword ,'with',method,'method. Now, you can easily download the result by clicking the button below')
     st.divider()
          
     #===show & download csv===
     tab1, tab2, tab3, tab4 = st.tabs(["📥 Result", "📥 List of Keywords", "📃 Reference", "📃 Recommended Reading"])
     
     with tab1:
         st.dataframe(keywords, use_container_width=True)
         @st.cache_data(ttl=3600)
         def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
         
         @st.cache_data(ttl=3600)
         def convert_txt(df):
             return df.to_csv(index=False, sep='\t', lineterminator='\r').encode('utf-8')
         
         if extype.endswith('.csv'):
             csv = convert_df(keywords)
             st.download_button(
                "Press to download result 👈",
                csv,
                "scopus.csv",
                "text/csv")
  
         elif extype.endswith('.txt'):
             keywords = rev_conv_txt()
             txt = convert_txt(keywords)
             st.download_button(
                "Press to download result 👈",
                txt,
                "savedrecs.txt",
                "text/csv")    
         
     with tab2:
         @st.cache_data(ttl=3600)
         def table_keyword():
             keytab = key.drop(['index'], axis=1).rename(columns={0: 'old'})
             return keytab
         keytab = table_keyword()
         st.dataframe(keytab, use_container_width=True)
                  
         @st.cache_data(ttl=3600)
         def convert_dfs(df):
             return df.to_csv(index=False).encode('utf-8')
                
         csv = convert_dfs(key)

         st.download_button(
             "Press to download keywords 👈",
             csv,
             "keywords.csv",
             "text/csv")
             
     with tab3:
         st.markdown('**Santosa, F. A. (2022). Prior steps into knowledge mapping: Text mining application and comparison. Issues in Science and Technology Librarianship, 102.** https://doi.org/10.29173/istl2736')
     
     with tab4:
         st.markdown('**Beri, A. (2021, January 27). Stemming vs Lemmatization. Medium.** https://towardsdatascience.com/stemming-vs-lemmatization-2daddabcb221')
         st.markdown('**Khyani, D., Siddhartha B S, Niveditha N M, &amp; Divya B M. (2020). An Interpretation of Lemmatization and Stemming in Natural Language Processing. Journal of University of Shanghai for Science and Technology , 22(10), 350–357.**  https://jusst.org/an-interpretation-of-lemmatization-and-stemming-in-natural-language-processing/')
         st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Text Pre-Processing. Text Mining for Information Professionals, 79–103.** https://doi.org/10.1007/978-3-030-85085-2_3')
