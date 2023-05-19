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
from ipywidgets.embed import embed_minimal_html
from nltk.stem.snowball import SnowballStemmer
from bertopic import BERTopic
import plotly.express as px
from sklearn.cluster import KMeans
import bitermplus as btm
import tmplot as tmp
import tomotopy


#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Topic Modeling")
st.subheader('Put your CSV file here ...')

#===optimize Biterm===
@st.cache_resource
def biterm_topic():
     topics_coords = tmp.prepare_coords(model)
     return topics_coords

def reset_data():
     st.cache_resource.clear()

#===upload file===
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    papers = pd.read_csv(uploaded_file)
    paper = papers.dropna(subset=['Abstract'])
    paper = paper[~paper.Abstract.str.contains("No abstract available")]
    paper = paper[~paper.Abstract.str.contains("STRAIT")]
        
    #===mapping===
    paper['Abstract_pre'] = paper['Abstract'].map(lambda x: re.sub('[,:;\.!?•-]', '', x))
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: x.lower())
    paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
        
    #===lemmatize===
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    paper['Abstract_lem'] = paper['Abstract_pre'].apply(lemmatize_words)
       
    #===stopword removal===
    stop = stopwords.words('english')
    paper['Abstract_stop'] = paper['Abstract_lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    method = st.selectbox(
            'Choose method',
            ('pyLDA', 'Biterm','BERTopic'))
        
    #===topic===
    if method is 'pyLDA':
         topic_abs = paper.Abstract_stop.values.tolist()
         topic_abs = [t.split(' ') for t in topic_abs]
         id2word = Dictionary(topic_abs)
         corpus = [id2word.doc2bow(text) for text in topic_abs]
         num_topic = st.slider('Choose number of topics', min_value=2, max_value=15, step=1)

         #===LDA===
         lda_model = LdaModel(corpus=corpus,
                     id2word=id2word,
                     num_topics=num_topic, 
                     random_state=0,
                     chunksize=100,
                     alpha='auto',
                     per_word_topics=True)

         pprint(lda_model.print_topics())
         doc_lda = lda_model[corpus]

         tab1, tab2, tab3 = st.tabs(["📈 Generate visualization & Calculate coherence", "📃 Reference", "📓 Recommended Reading"])

         with tab1:
         #===visualization===
             with st.spinner('Calculating and Creating pyLDAvis Visualization ...'):
                 coherence_model_lda = CoherenceModel(model=lda_model, texts=topic_abs, dictionary=id2word, coherence='c_v')
                 coherence_lda = coherence_model_lda.get_coherence()
                 st.write('Score: ', (coherence_lda))
                 vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
                 py_lda_vis_html = pyLDAvis.prepared_data_to_html(vis)
                 components.html(py_lda_vis_html, width=1700, height=800)
                 st.markdown('Copyright (c) 2015, Ben Mabey. https://github.com/bmabey/pyLDAvis')

         with tab2:
             st.markdown('**Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing and interpreting topics. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces.** https://doi.org/10.3115/v1/w14-3110')

         with tab3:
             st.markdown('**Chen, X., & Wang, H. (2019, January). Automated chat transcript analysis using topic modeling for library reference services. Proceedings of the Association for Information Science and Technology, 56(1), 368–371.** https://doi.org/10.1002/pra2.31')
             st.markdown('**Joo, S., Ingram, E., & Cahill, M. (2021, December 15). Exploring Topics and Genres in Storytime Books: A Text Mining Approach. Evidence Based Library and Information Practice, 16(4), 41–62.** https://doi.org/10.18438/eblip29963')
             st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Topic Modeling. Text Mining for Information Professionals, 105–137.** https://doi.org/10.1007/978-3-030-85085-2_4')
             st.markdown('**Lamba, M., & Madhusudhan, M. (2019, June 7). Mapping of topics in DESIDOC Journal of Library and Information Technology, India: a study. Scientometrics, 120(2), 477–505.** https://doi.org/10.1007/s11192-019-03137-5')
     
     #===Biterm===
    elif method is 'Biterm':
        topic_abs = paper.Abstract_stop.values.tolist()
        num_bitopic = st.slider('Choose number of topics', min_value=2, max_value=20, step=1, on_change=reset_data)
        topic_abs = paper.Abstract_stop.values.tolist()       
        X, vocabulary, vocab_dict = btm.get_words_freqs(topic_abs)
        tf = np.array(X.sum(axis=0)).ravel()
        docs_vec = btm.get_vectorized_docs(topic_abs, vocabulary)
        docs_lens = list(map(len, docs_vec))
        biterms = btm.get_biterms(docs_vec)
        model = btm.BTM(
          X, vocabulary, seed=12321, T=num_bitopic, M=20, alpha=50/8, beta=0.01)
        model.fit(biterms, iterations=20)
        p_zd = model.transform(docs_vec)
        coherence = model.coherence_
        phi = tmp.get_phi(model)
        try:
          topik = biterm_topic()
          totaltop = topik.label.values.tolist()
          with st.spinner('Visualizing, please wait ....'):          
             tab1, tab2 = st.tabs(["📈 Generate visualization", "📃 Reference"])
             with tab1:
                  col1, col2 = st.columns(2)
                  with col1:
                    num_bitopic_vis = st.selectbox(
                         'Choose topic',
                         (totaltop))
                    btmvis_coords = tmp.plot_scatter_topics(topik, size_col='size', label_col='label', topic=num_bitopic_vis)
                    st.altair_chart(btmvis_coords, use_container_width=True)
                  with col2:
                    terms_probs = tmp.calc_terms_probs_ratio(phi, topic=num_bitopic_vis, lambda_=1)
                    btmvis_probs = tmp.plot_terms(terms_probs, font_size=12)
                    st.altair_chart(btmvis_probs, use_container_width=True)
             with tab2: 
                    st.markdown('**Yan, X., Guo, J., Lan, Y., & Cheng, X. (2013, May 13). A biterm topic model for short texts. Proceedings of the 22nd International Conference on World Wide Web.** https://doi.org/10.1145/2488388.2488514')
          
        except ValueError:
          st.error('Please raise the number of topics')
    
     #===BERTopic===
    elif method is 'BERTopic':
        num_btopic = st.slider('Choose number of topics', min_value=4, max_value=20, step=1)
        topic_abs = paper.Abstract_stop.values.tolist()
        topic_time = paper.Year.values.tolist()
        cluster_model = KMeans(n_clusters=num_btopic)
        topic_model = BERTopic(hdbscan_model=cluster_model).fit(topic_abs)
        topics, probs = topic_model.fit_transform(topic_abs)
        
        tab1, tab2 = st.tabs(["📈 Generate visualization", "📃 Reference"])
        with tab1:
          #===visualization===
          viz = st.selectbox(
            'Choose visualization',
            ('Visualize Topics', 'Visualize Documents', 'Visualize Document Hierarchy', 'Visualize Topic Similarity', 'Visualize Terms', 'Visualize Topics over Time'))

          if viz == 'Visualize Topics':
                 fig1 = topic_model.visualize_topics()
                 st.write(fig1)

          elif viz == 'Visualize Documents':
                 fig2 = topic_model.visualize_documents(topic_abs)
                 st.write(fig2)

          elif viz == 'Visualize Document Hierarchy':
                 fig3 = topic_model.visualize_hierarchy(top_n_topics=num_btopic)
                 st.write(fig3)

          elif viz == 'Visualize Topic Similarity':
                 fig4 = topic_model.visualize_heatmap(n_clusters=num_btopic-1, width=1000, height=1000)
                 st.write(fig4)

          elif viz == 'Visualize Terms':
                 fig5 = topic_model.visualize_barchart(top_n_topics=num_btopic)
                 st.write(fig5)

          elif viz == 'Visualize Topics over Time':
                 topics_over_time = topic_model.topics_over_time(topic_abs, topic_time)
                 fig6 = topic_model.visualize_topics_over_time(topics_over_time)
                 st.write(fig6)
                    
        with tab2:
          st.markdown('**Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.** https://doi.org/10.48550/arXiv.2203.05794')
