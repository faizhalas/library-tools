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
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
            <style>
            #MainMenu 
            {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="collapsedControl"] {display: none}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.popover("🔗 Menu"):
    st.page_link("Home.py", label="Home", icon="🏠")
    st.page_link("pages/1 Scattertext.py", label="Scattertext", icon="1️⃣")
    st.page_link("pages/2 Topic Modeling.py", label="Topic Modeling", icon="2️⃣")
    st.page_link("pages/3 Bidirected Network.py", label="Bidirected Network", icon="3️⃣")
    st.page_link("pages/4 Sunburst.py", label="Sunburst", icon="4️⃣")
    st.page_link("pages/5 Burst Detection.py", label="Burst Detection", icon="5️⃣")
    st.page_link("pages/6 Keywords Stem.py", label="Keywords Stem", icon="6️⃣")
    

st.header("Keywords Stem", anchor=False)
st.subheader('Put your file here...', anchor=False)

def reset_data():
     st.cache_data.clear()

#===check filetype===
@st.cache_data(ttl=3600)
def get_ext(extype):
    extype = uploaded_file.name
    return extype
     
#===upload===
@st.cache_data(ttl=3600)
def upload(extype):
    keywords = pd.read_csv(uploaded_file)
    return keywords

@st.cache_data(ttl=3600)
def conv_txt(extype):
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DE': 'Author Keywords',
            'ID': 'Keywords Plus'}
    keywords = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
    keywords.rename(columns=col_dict, inplace=True)
    return keywords

@st.cache_data(ttl=3600)
def rev_conv_txt(extype):
    col_dict_rev = {'Title': 'TI',
            'Source title': 'SO',
            'Author Keywords': 'DE',
            'Keywords Plus': 'ID'}
    keywords.rename(columns=col_dict_rev, inplace=True)
    return keywords

@st.cache_data(ttl=3600)
def get_data(extype):
    list_of_column_key = list(keywords.columns)
    list_of_column_key = [k for k in list_of_column_key if 'Keyword' in k]
    return list_of_column_key

uploaded_file = st.file_uploader('', type=['csv','txt'], on_change=reset_data)

if uploaded_file is not None:
     extype = get_ext(uploaded_file)
     if extype.endswith('.csv'):
         keywords = upload(extype) 
                  
     elif extype.endswith('.txt'):
         keywords = conv_txt(extype)
         
     list_of_column_key = get_data(extype)

     col1, col2 = st.columns(2)
     with col1:
        method = st.selectbox(
             'Choose method',
           ('Lemmatization', 'Stemming'), on_change=reset_data)
     with col2:
        keyword = st.selectbox(
            'Choose column',
           (list_of_column_key), on_change=reset_data)

     @st.cache_data(ttl=3600)
     def clean_keyword(extype):      
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
     def Lemmatization(extype):
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
     def Stemming(extype):
        stemmer = SnowballStemmer("english")
        def stem_words(text):
            words = text.split()
            words = [stemmer.stem(word) for word in words]
            return ' '.join(words)
        keywords[keyword] = keywords[keyword].apply(stem_words)
        key['new'] = key['new'].apply(stem_words)
        keywords[keyword] = keywords[keyword].map(lambda x: re.sub(' ; ', '; ', x))
        return keywords, key
     
     keywords, key = clean_keyword(extype) 
     
     if method is 'Lemmatization':
         keywords, key = Lemmatization(extype)
     else:
         keywords, key = Stemming(extype)
            
     st.write('Congratulations! 🤩 You choose',keyword ,'with',method,'method. Now, you can easily download the result by clicking the button below')
     st.divider()
          
     #===show & download csv===
     tab1, tab2, tab3, tab4 = st.tabs(["📥 Result", "📥 List of Keywords", "📃 Reference", "📃 Recommended Reading"])
     
     with tab1:
         st.dataframe(keywords, use_container_width=True, hide_index=True)
         @st.cache_data(ttl=3600)
         def convert_df(extype):
            return keywords.to_csv(index=False).encode('utf-8')
         
         @st.cache_data(ttl=3600)
         def convert_txt(extype):
             return keywords.to_csv(index=False, sep='\t', lineterminator='\r').encode('utf-8')
         
         if extype.endswith('.csv'):
             csv = convert_df(extype)
             st.download_button(
                "Press to download result 👈",
                csv,
                "scopus.csv",
                "text/csv")
  
         elif extype.endswith('.txt'):
             keywords = rev_conv_txt(extype)
             txt = convert_txt(extype)
             st.download_button(
                "Press to download result 👈",
                txt,
                "savedrecs.txt",
                "text/csv")    
         
     with tab2:
         @st.cache_data(ttl=3600)
         def table_keyword(extype):
             keytab = key.drop(['index'], axis=1).rename(columns={0: 'label'})
             return keytab
         #===coloring the same keywords===
         @st.cache_data(ttl=3600)
         def highlight_cells(value):
             if keytab['new'].duplicated(keep=False).any() and keytab['new'].duplicated(keep=False)[keytab['new'] == value].any():
                 return 'background-color: yellow'
             return '' 
         keytab = table_keyword(extype) 
         st.dataframe(keytab.style.applymap(highlight_cells, subset=['new']), use_container_width=True, hide_index=True)
                  
         @st.cache_data(ttl=3600)
         def convert_dfs(extype):
             return key.to_csv(index=False).encode('utf-8')
                
         csv = convert_dfs(extype)

         st.download_button(
             "Press to download keywords 👈",
             csv,
             "keywords.csv",
             "text/csv")
             
     with tab3:
         st.markdown('**Santosa, F. A. (2023). Prior steps into knowledge mapping: Text mining application and comparison. Issues in Science and Technology Librarianship, 102.** https://doi.org/10.29173/istl2736')
     
     with tab4:
         st.markdown('**Beri, A. (2021, January 27). Stemming vs Lemmatization. Medium.** https://towardsdatascience.com/stemming-vs-lemmatization-2daddabcb221')
         st.markdown('**Khyani, D., Siddhartha B S, Niveditha N M, &amp; Divya B M. (2020). An Interpretation of Lemmatization and Stemming in Natural Language Processing. Journal of University of Shanghai for Science and Technology , 22(10), 350–357.**  https://jusst.org/an-interpretation-of-lemmatization-and-stemming-in-natural-language-processing/')
         st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Text Pre-Processing. Text Mining for Information Professionals, 79–103.** https://doi.org/10.1007/978-3-030-85085-2_3')
