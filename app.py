#import module
import random

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis.gensim_models
import regex
import seaborn as sns
import streamlit as st
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import streamlit.components.v1 as components
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from umap import UMAP
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import plotly.express as px
import re


#title
st.title('Faizhal App')

#Header
st.header('LDA')

#subhead
st.subheader('Silahkan masukan CSV File')

#Upload
from io import StringIO

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

    #LDA
    lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=2, #num of topic
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    #coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=topic_abs, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    st.write(coherence_lda)


