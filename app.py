#import module
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


st.set_page_config(layout="wide")

#title
st.title('Coconut Library Tools')

#page
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Keywords Stem", "Topic Modelling", "About"])

with tab1:
   st.header("How to use")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)



with tab2:
   st.header("Keywords Stem")
   
   #subhead
   st.subheader('Put your CSV file and choose method')

   file_key = st.file_uploader("Choose your a file")

   method = st.selectbox(
       'Choose method',
      ('Stemming', 'Lemmatization'))
   
   keyword = st.selectbox(
       'Choose column',
      ('Author Keywords', 'Index Keywords'))


   if file_key is not None:
        papers = pd.read_csv(file_key)
        keywords = papers.dropna(subset=[keyword])
        datakey = keywords[keyword].map(lambda x: re.sub(' ', '_', x))
        datakey = datakey.map(lambda x: re.sub(';_', ' ', x))
        datakey = datakey.map(lambda x: x.lower())
        if method is 'Lemmatization':
             lemmatizer = WordNetLemmatizer()
             def lemmatize_words(text):
                words = text.split()
                words = [lemmatizer.lemmatize(word,pos='v') for word in words]
                return ' '.join(words)
             datakey = datakey.apply(lemmatize_words)
             #st.write(datakey)
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
        
        def convert_df(df):
           return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(keywords)
        st.download_button(
            "Press to Download 👈",
            csv,
            "scopus.csv",
            "text/csv",
            key='download-csv')









with tab3:
   st.header("pyLDA")
   
   #subhead
   st.subheader('Put your CSV file and click generate')

   uploaded_file = st.file_uploader("Choose a file")
   if uploaded_file is not None:
       # Can be used wherever a "file-like" object is accepted:
       papers = pd.read_csv(uploaded_file)
       #st.dataframe(scopus.style.where(lambda val: '[No abstract available]' in str(val), 'color: red', subset=['Abstract']))
       paper = papers.dropna(subset=['Abstract'])
       paper = paper[~paper.Abstract.str.contains("No abstract available")]
       paper = paper[~paper.Abstract.str.contains("STRAIT")]
        
       #mapping
       paper['Abstract_pre'] = paper['Abstract'].map(lambda x: re.sub('[,:;\.!?•-]', '', x))
       paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: x.lower())
       paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
        
       #lemmatize
       lemmatizer = WordNetLemmatizer()
       def lemmatize_words(text):
           words = text.split()
           words = [lemmatizer.lemmatize(word,pos='v') for word in words]
           return ' '.join(words)
       paper['Abstract_lem'] = paper['Abstract_pre'].apply(lemmatize_words)
       
       #stopword removal
       stop = stopwords.words('english')
       paper['Abstract_stop'] = paper['Abstract_lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        
       #topic
       topic_abs = paper.Abstract_stop.values.tolist()
       topic_abs = [t.split(' ') for t in topic_abs]
       id2word = Dictionary(topic_abs)
       corpus = [id2word.doc2bow(text) for text in topic_abs]

       num_topic = st.slider('Choose number of topics', min_value=2, max_value=15, step=1)
       #LDA
       lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=num_topic, #num of topic
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

       pprint(lda_model.print_topics())
       doc_lda = lda_model[corpus]

   #coherence score
   if st.button('📐 Calculate coherence'):
               with st.spinner('Calculating, please wait ....'):    
                  coherence_model_lda = CoherenceModel(model=lda_model, texts=topic_abs, dictionary=id2word, coherence='c_v')
                  coherence_lda = coherence_model_lda.get_coherence()
                  st.write(coherence_lda)

   if st.button('📈 Generate visualization'):
               with st.spinner('Creating pyLDAvis Visualization ...'):
                   vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
                   py_lda_vis_html = pyLDAvis.prepared_data_to_html(vis)
                   components.html(py_lda_vis_html, width=1700, height=800)
                   st.markdown('👍 find out https://github.com/bmabey/pyLDAvis')

with tab4:
   st.header("About")
   st.subheader('Contributors')
   st.write('Faizhal Arif Santosa, (you can be a part of this app)')

   st.subheader('Involving')
   st.write('If you interested to develop this program, please contact me')   

   st.subheader('Citation')
   st.write('If you interested to develop this program, please contact me') 
