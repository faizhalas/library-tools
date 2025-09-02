#import module
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re
import string
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
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from pprint import pprint
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from io import StringIO
from ipywidgets.embed import embed_minimal_html
from nltk.stem.snowball import SnowballStemmer
from bertopic import BERTopic
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import bitermplus as btm
import tmplot as tmp
import tomotopy
import sys
import spacy
import en_core_web_sm
import pipeline
from html2image import Html2Image
from umap import UMAP
import os
import time
import json
from tools import sourceformat as sf
import datamapplot
from sentence_transformers import SentenceTransformer

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
    st.page_link("https://www.coconut-libtool.com/", label="Home", icon="🏠")
    st.page_link("pages/1 Scattertext.py", label="Scattertext", icon="1️⃣")
    st.page_link("pages/2 Topic Modeling.py", label="Topic Modeling", icon="2️⃣")
    st.page_link("pages/3 Bidirected Network.py", label="Bidirected Network", icon="3️⃣")
    st.page_link("pages/4 Sunburst.py", label="Sunburst", icon="4️⃣")
    st.page_link("pages/5 Burst Detection.py", label="Burst Detection", icon="5️⃣")
    st.page_link("pages/6 Keywords Stem.py", label="Keywords Stem", icon="6️⃣")
    st.page_link("pages/7 Sentiment Analysis.py", label="Sentiment Analysis", icon="7️⃣")
    st.page_link("pages/8 Shifterator.py", label="Shifterator", icon="8️⃣")

st.header("Topic Modeling", anchor=False)
st.subheader('Put your file here...', anchor=False)

#========unique id========
@st.cache_resource(ttl=3600)
def create_list():
    l = [1, 2, 3]
    return l

l = create_list()
first_list_value = l[0]
l[0] = first_list_value + 1
uID = str(l[0])

@st.cache_data(ttl=3600)
def get_ext(uploaded_file):
    extype = uID+uploaded_file.name
    return extype

#===clear cache===

def reset_biterm():
    try:
        biterm_map.clear()
        biterm_bar.clear()
    except NameError:
        biterm_topic.clear()

def reset_all():
    st.cache_data.clear()

#===avoiding deadlock===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
#===upload file===
@st.cache_data(ttl=3600)
def upload(file):
    papers = pd.read_csv(uploaded_file)
    if "dimensions" in uploaded_file.name.lower():
        papers = sf.dim(papers)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        papers.rename(columns=col_dict, inplace=True)
    return papers

@st.cache_data(ttl=3600)
def conv_txt(extype):
    if("pmc" in uploaded_file.name.lower() or "pubmed" in uploaded_file.name.lower()):
        file = uploaded_file
        papers = sf.medline(file)

    elif("hathi" in uploaded_file.name.lower()):
        papers = pd.read_csv(uploaded_file,sep = '\t')
        papers = sf.htrc(papers)
        col_dict={'title': 'title',
        'rights_date_used': 'Year',
        }
        papers.rename(columns=col_dict, inplace=True)
        
    else:
        col_dict = {'TI': 'Title',
                'SO': 'Source title',
                'DE': 'Author Keywords',
                'DT': 'Document Type',
                'AB': 'Abstract',
                'TC': 'Cited by',
                'PY': 'Year',
                'ID': 'Keywords Plus'}
        papers = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
        papers.rename(columns=col_dict, inplace=True)
    print(papers)
    return papers


@st.cache_data(ttl=3600)
def conv_json(extype):
    col_dict={'title': 'title',
    'rights_date_used': 'Year',
    }

    data = json.load(uploaded_file)
    hathifile = data['gathers']
    keywords = pd.DataFrame.from_records(hathifile)
    
    keywords = sf.htrc(keywords)
    keywords.rename(columns=col_dict,inplace=True)
    return keywords

@st.cache_data(ttl=3600)
def conv_pub(extype):
    if (get_ext(extype)).endswith('.tar.gz'):
        bytedata = extype.read()
        keywords = sf.readPub(bytedata)
    elif (get_ext(extype)).endswith('.xml'):
        bytedata = extype.read()
        keywords = sf.readxml(bytedata)
    return keywords

#===Read data===
uploaded_file = st.file_uploader('', type=['csv', 'txt','json','tar.gz','xml'], on_change=reset_all)

if uploaded_file is not None:
    try:
        extype = get_ext(uploaded_file)
    
        if extype.endswith('.csv'):
             papers = upload(extype) 
        elif extype.endswith('.txt'):
             papers = conv_txt(extype)

        elif extype.endswith('.json'):
            papers = conv_json(extype)
        elif extype.endswith('.tar.gz') or extype.endswith('.xml'):
            papers = conv_pub(uploaded_file)

        coldf = sorted(papers.select_dtypes(include=['object']).columns.tolist())
            
        c1, c2, c3 = st.columns([3,3,4])
        method = c1.selectbox(
                'Choose method',
                ('Choose...', 'pyLDA', 'Biterm', 'BERTopic'))
        ColCho = c2.selectbox('Choose column', (coldf))
        num_cho = c3.number_input('Choose number of topics', min_value=2, max_value=30, value=5)

        d1, d2 = st.columns([3,7])
        xgram = d1.selectbox("N-grams", ("1", "2", "3"), on_change=reset_all)
        xgram = int(xgram)
        words_to_remove = d2.text_input("Remove specific words. Separate words by semicolons (;)", on_change=reset_all)
    
        rem_copyright = d1.toggle('Remove copyright statement', value=True, on_change=reset_all)
        rem_punc = d2.toggle('Remove punctuation', value=True, on_change=reset_all)

        #===advance settings===
        with st.expander("🧮 Show advance settings"): 
            t1, t2, t3, t4 = st.columns(4)
            if method == 'pyLDA':
                py_random_state = t1.number_input('Random state', min_value=0, max_value=None, step=1, help='Ensuring the reproducibility.')
                py_chunksize = t2.number_input('Chunk size', value=100 , min_value=10, max_value=None, step=1, help='Number of documents to be used in each training chunk.')
                opt_threshold = t3.number_input('Threshold (Gensim)', value=100 , min_value=1, max_value=None, step=1, help='Lower = More phrases. Higher = Fewer phrases.')
                opt_relevance = t4.number_input('Lambda (λ)', value=0.6 , min_value=0.0, max_value=1.0, step=0.01, help='Lower = More unique. Higher = More frequent.')
                
                
            elif method == 'Biterm':
                btm_seed = t1.number_input('Random state seed', value=100 , min_value=1, max_value=None, step=1, help='Ensuring the reproducibility.')
                btm_iterations = t2.number_input('Iterations number', value=20 , min_value=2, max_value=None, step=1, help='Number of iterations the model fitting process has gone through.')
                opt_threshold = t3.number_input('Threshold (Gensim)', value=100 , min_value=1, max_value=None, step=1, help='Lower = More phrases. Higher = Fewer phrases.')
                
            elif method == 'BERTopic':
                #u1, u2 = st.columns([5,5])
                
                bert_top_n_words = t1.number_input('top_n_words', value=5 , min_value=5, max_value=25, step=1, help='Number of words per topic.')
                bert_random_state = t2.number_input('random_state', value=42 , min_value=1, max_value=None, step=1, help="Please be aware we currently can't do the reproducibility on Bertopic.")
                bert_n_components = t3.number_input('n_components', value=5 , min_value=1, max_value=None, step=1, help='The dimensionality of the embeddings after reducing them.')
                bert_n_neighbors = t4.number_input('n_neighbors', value=15 , min_value=1, max_value=None, step=1, help='The number of neighboring sample points used when making the manifold approximation.')
                bert_embedding_model = st.radio(
                    "embedding_model", 
                    ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "en_core_web_sm"], index=0, horizontal=True, help= 'Select paraphrase-multilingual if your documents are in a language other than English or are multilingual.')
            else:
                st.write('Please choose your preferred method')
        
        #===clean csv===
        @st.cache_data(ttl=3600, show_spinner=False)
        def clean_csv(extype):
            paper = papers.dropna(subset=[ColCho])
                     
            #===mapping===
            paper['Abstract_pre'] = paper[ColCho].map(lambda x: x.lower())
            if rem_punc:
                paper['Abstract_pre'] = paper['Abstract_pre'].map(
                    lambda x: re.sub(f"[{re.escape(string.punctuation)}]", " ", x)
                ).map(lambda x: re.sub(r"\s+", " ", x).strip())
                paper['Abstract_pre'] = paper['Abstract_pre'].str.replace('[\u2018\u2019\u201c\u201d]', '', regex=True)
            if rem_copyright:  
                paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
            
            #===stopword removal===
            stop = stopwords.words('english')
            paper['Abstract_stop'] = paper['Abstract_pre'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
              
            #===lemmatize===
            lemmatizer = WordNetLemmatizer()
            
            @st.cache_data(ttl=3600)
            def lemmatize_words(text):
                words = text.split()
                words = [lemmatizer.lemmatize(word) for word in words]
                return ' '.join(words)
            paper['Abstract_lem'] = paper['Abstract_stop'].apply(lemmatize_words)
        
            words_rmv = [word.strip() for word in words_to_remove.split(";")]
            remove_dict = {word: None for word in words_rmv}
            
            @st.cache_data(ttl=3600)
            def remove_words(text):
                 words = text.split()
                 cleaned_words = [word for word in words if word not in remove_dict]
                 return ' '.join(cleaned_words) 
            paper['Abstract_lem'] = paper['Abstract_lem'].map(remove_words)
             
            topic_abs = paper.Abstract_lem.values.tolist()
            return topic_abs, paper
    
        topic_abs, paper=clean_csv(extype) 
                 
        if st.button("Submit", on_click=reset_all):
            num_topic = num_cho  
    
        if method == 'BERTopic':
            st.info('BERTopic is an expensive process when dealing with a large volume of text with our existing resources. Please kindly wait until the visualization appears.', icon="ℹ️")
               
        #===topic===
        if method == 'Choose...':
            st.write('')
    
        elif method == 'pyLDA':       
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help"])
    
            with tab1:
            #===visualization===
                @st.cache_data(ttl=3600, show_spinner=False)
                def pylda(extype):
                    topic_abs_LDA = [t.split(' ') for t in topic_abs]

                    bigram = Phrases(topic_abs_LDA, min_count=xgram, threshold=opt_threshold)
                    trigram = Phrases(bigram[topic_abs_LDA], threshold=opt_threshold)
                    bigram_mod = Phraser(bigram)
                    trigram_mod = Phraser(trigram)
                    
                    topic_abs_LDA = [trigram_mod[bigram_mod[doc]] for doc in topic_abs_LDA]

                    id2word = Dictionary(topic_abs_LDA)
                    corpus = [id2word.doc2bow(text) for text in topic_abs_LDA]
                    #===LDA===
                    lda_model = LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=num_topic, 
                                random_state=py_random_state,
                                chunksize=py_chunksize,
                                alpha='auto',
                                gamma_threshold=opt_relevance,
                                per_word_topics=False)
                    pprint(lda_model.print_topics())
                    doc_lda = lda_model[corpus]
                    topics = lda_model.show_topics(num_words = 30,formatted=False)

                    #===visualization===
                    coherence_model_lda = CoherenceModel(model=lda_model, texts=topic_abs_LDA, dictionary=id2word, coherence='c_v')
                    coherence_lda = coherence_model_lda.get_coherence()
                    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
                    py_lda_vis_html = pyLDAvis.prepared_data_to_html(vis)
                    return py_lda_vis_html, coherence_lda, vis, topics
                       
                with st.spinner('Performing computations. Please wait ...'):
                    try:
                        py_lda_vis_html, coherence_lda, vis, topics = pylda(extype)
                        st.write('Coherence score: ', coherence_lda)
                        components.html(py_lda_vis_html, width=1500, height=800)
                        st.markdown('Copyright (c) 2015, Ben Mabey. https://github.com/bmabey/pyLDAvis')
                           
                        @st.cache_data(ttl=3600, show_spinner=False)
                        def img_lda(vis):
                            pyLDAvis.save_html(vis, 'output.html')
                            hti = Html2Image()
                            hti.browser.flags = ['--default-background-color=ffffff', '--hide-scrollbars']
                            hti.browser.use_new_headless = None
                            css = "body {background: white;}"
                            hti.screenshot( 
                                other_file='output.html', css_str=css, size=(1500, 800),
                                save_as='ldavis_img.png'
                            )

                        img_lda(vis)
                        
                        d1, d2 = st.columns(2)
                        with open("ldavis_img.png", "rb") as file:
                            btn = d1.download_button(
                                label="Download image",
                                data=file,
                                file_name="ldavis_img.png",
                                mime="image/png"
                                )
                        
                        #===download results===#
                        resultf = pd.DataFrame(topics)
                        #formatting
                        resultf = resultf.transpose()
                        resultf = resultf.drop([0])
                        resultf = resultf.explode(list(range(len(resultf.columns))), ignore_index=False)
                        
                        resultcsv = resultf.to_csv().encode("utf-8")
                        d2.download_button(
                            label = "Download Results",
                            data=resultcsv,
                            file_name="results.csv",
                            mime="text\csv",
                            on_click="ignore")

                    except NameError as f:
                        st.warning('🖱️ Please click Submit')
    
            with tab2:
                st.markdown('**Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing and interpreting topics. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces.** https://doi.org/10.3115/v1/w14-3110')
    
            with tab3:
                st.markdown('**Chen, X., & Wang, H. (2019, January). Automated chat transcript analysis using topic modeling for library reference services. Proceedings of the Association for Information Science and Technology, 56(1), 368–371.** https://doi.org/10.1002/pra2.31')
                st.markdown('**Joo, S., Ingram, E., & Cahill, M. (2021, December 15). Exploring Topics and Genres in Storytime Books: A Text Mining Approach. Evidence Based Library and Information Practice, 16(4), 41–62.** https://doi.org/10.18438/eblip29963')
                st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Topic Modeling. Text Mining for Information Professionals, 105–137.** https://doi.org/10.1007/978-3-030-85085-2_4')
                st.markdown('**Lamba, M., & Madhusudhan, M. (2019, June 7). Mapping of topics in DESIDOC Journal of Library and Information Technology, India: a study. Scientometrics, 120(2), 477–505.** https://doi.org/10.1007/s11192-019-03137-5')
         
            with tab4:
                st.subheader(':blue[pyLDA]', anchor=False)
                st.button('Download image')
                st.text("Click Download Image button.")
                st.divider()
                st.subheader(':blue[Downloading CSV Results]', anchor=False)
                st.button("Download Results")
                st.text("Click Download results button at bottom of page")

         #===Biterm===
        elif method == 'Biterm':            
                 
            #===optimize Biterm===
            @st.cache_data(ttl=3600, show_spinner=False)
            def biterm_topic(extype):
                tokenized_abs = [t.split(' ') for t in topic_abs]

                bigram = Phrases(tokenized_abs, min_count=xgram, threshold=opt_threshold)
                trigram = Phrases(bigram[tokenized_abs], threshold=opt_threshold)
                bigram_mod = Phraser(bigram)
                trigram_mod = Phraser(trigram)
            
                topic_abs_ngram = [trigram_mod[bigram_mod[doc]] for doc in tokenized_abs]
            
                topic_abs_str = [' '.join(doc) for doc in topic_abs_ngram]

                
                X, vocabulary, vocab_dict = btm.get_words_freqs(topic_abs_str)
                tf = np.array(X.sum(axis=0)).ravel()
                docs_vec = btm.get_vectorized_docs(topic_abs, vocabulary)
                docs_lens = list(map(len, docs_vec))
                biterms = btm.get_biterms(docs_vec)
                
                model = btm.BTM(X, vocabulary, seed=btm_seed, T=num_topic, M=20, alpha=50/8, beta=0.01)
                model.fit(biterms, iterations=btm_iterations)
                
                p_zd = model.transform(docs_vec)
                coherence = model.coherence_
                phi = tmp.get_phi(model)
                topics_coords = tmp.prepare_coords(model)
                totaltop = topics_coords.label.values.tolist()
                perplexity = model.perplexity_
                top_topics = model.df_words_topics_
                
                return topics_coords, phi, totaltop, perplexity, top_topics
    
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help"])
            with tab1:
                try:
                    with st.spinner('Performing computations. Please wait ...'): 
                        topics_coords, phi, totaltop, perplexity, top_topics = biterm_topic(extype)            
                        col1, col2 = st.columns([4,6])
                      
                        @st.cache_data(ttl=3600)
                        def biterm_map(extype):
                            btmvis_coords = tmp.plot_scatter_topics(topics_coords, size_col='size', label_col='label', topic=numvis)
                            return btmvis_coords
                                
                        @st.cache_data(ttl=3600)
                        def biterm_bar(extype):
                            terms_probs = tmp.calc_terms_probs_ratio(phi, topic=numvis, lambda_=1)
                            btmvis_probs = tmp.plot_terms(terms_probs, font_size=12)
                            return btmvis_probs
                                
                        with col1:
                            st.write('Perplexity score: ', perplexity)
                            st.write('')
                            numvis = st.selectbox(
                                'Choose topic',
                                (totaltop), on_change=reset_biterm)
                            btmvis_coords = biterm_map(extype)
                            st.altair_chart(btmvis_coords)
                        with col2:
                            btmvis_probs = biterm_bar(extype)
                            st.altair_chart(btmvis_probs, use_container_width=True)
    
                        #===download results===#
                        resultcsv = top_topics.to_csv().encode("utf-8")
                        st.download_button(label = "Download Results", data=resultcsv, file_name="results.csv", mime="text\csv", on_click="ignore")

                except ValueError as g:
                    st.error('🙇‍♂️ Please raise the number of topics and click submit')
                    
                except NameError as f:
                    st.warning('🖱️ Please click Submit')
    
            with tab2: 
                st.markdown('**Yan, X., Guo, J., Lan, Y., & Cheng, X. (2013, May 13). A biterm topic model for short texts. Proceedings of the 22nd International Conference on World Wide Web.** https://doi.org/10.1145/2488388.2488514')
            with tab3:
                st.markdown('**Cai, M., Shah, N., Li, J., Chen, W. H., Cuomo, R. E., Obradovich, N., & Mackey, T. K. (2020, August 26). Identification and characterization of tweets related to the 2015 Indiana HIV outbreak: A retrospective infoveillance study. PLOS ONE, 15(8), e0235150.** https://doi.org/10.1371/journal.pone.0235150')
                st.markdown('**Chen, Y., Dong, T., Ban, Q., & Li, Y. (2021). What Concerns Consumers about Hypertension? A Comparison between the Online Health Community and the Q&A Forum. International Journal of Computational Intelligence Systems, 14(1), 734.** https://doi.org/10.2991/ijcis.d.210203.002')
                st.markdown('**George, Crissandra J., "AMBIGUOUS APPALACHIANNESS: A LINGUISTIC AND PERCEPTUAL INVESTIGATION INTO ARC-LABELED PENNSYLVANIA COUNTIES" (2022). Theses and Dissertations-- Linguistics. 48.** https://doi.org/10.13023/etd.2022.217')
                st.markdown('**Li, J., Chen, W. H., Xu, Q., Shah, N., Kohler, J. C., & Mackey, T. K. (2020). Detection of self-reported experiences with corruption on twitter using unsupervised machine learning. Social Sciences & Humanities Open, 2(1), 100060.** https://doi.org/10.1016/j.ssaho.2020.100060')
            with tab4:
                st.subheader(':blue[Biterm]', anchor=False)
                st.text("Click the three dots at the top right then select the desired format.")
                st.markdown("![Downloading visualization](https://raw.githubusercontent.com/faizhalas/library-tools/main/images/download_biterm.jpg)")  
                st.divider()
                st.subheader(':blue[Downloading CSV Results]', anchor=False)
                st.button("Download Results")
                st.text("Click Download results button at bottom of page")


         #===BERTopic===
        elif method == 'BERTopic':
            @st.cache_data(ttl=3600, show_spinner=False)
            def bertopic_vis(extype):
                umap_model = UMAP(n_neighbors=bert_n_neighbors, n_components=bert_n_components, 
                    min_dist=0.0, metric='cosine', random_state=bert_random_state)   
                cluster_model = KMeans(n_clusters=num_topic)
                if bert_embedding_model == 'all-MiniLM-L6-v2':
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    lang = 'en'
                    embeddings = model.encode(topic_abs, show_progress_bar=True)
                    
                elif bert_embedding_model == 'en_core_web_sm':
                    nlp = en_core_web_sm.load(exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
                    lang = 'en'
                    embeddings = np.array([nlp(text).vector for text in topic_abs])
                    
                elif bert_embedding_model == 'paraphrase-multilingual-MiniLM-L12-v2':
                    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    lang = 'multilingual'
                    embeddings = model.encode(topic_abs, show_progress_bar=True)

                vectorizer_model = CountVectorizer(ngram_range=(1, xgram), stop_words='english')
                topic_model = BERTopic(embedding_model=None, hdbscan_model=cluster_model, language=lang, umap_model=umap_model, vectorizer_model=vectorizer_model, top_n_words=bert_top_n_words)
                topics, probs = topic_model.fit_transform(topic_abs, embeddings=embeddings)
                return topic_model, topics, probs, embeddings
            
            @st.cache_data(ttl=3600, show_spinner=False)
            def Vis_Topics(extype):
                fig1 = topic_model.visualize_topics()
                return fig1
            
            @st.cache_data(ttl=3600, show_spinner=False)
            def Vis_Documents(extype):
                fig2 = topic_model.visualize_document_datamap(topic_abs, embeddings=embeddings)
                return fig2
    
            @st.cache_data(ttl=3600, show_spinner=False)
            def Vis_Hierarchy(extype):
                fig3 = topic_model.visualize_hierarchy(top_n_topics=num_topic)
                return fig3
        
            @st.cache_data(ttl=3600, show_spinner=False)
            def Vis_Heatmap(extype):
                global topic_model
                fig4 = topic_model.visualize_heatmap(n_clusters=num_topic-1, width=1000, height=1000)
                return fig4
    
            @st.cache_data(ttl=3600, show_spinner=False)
            def Vis_Barchart(extype):
                fig5 = topic_model.visualize_barchart(top_n_topics=num_topic)
                return fig5
           
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help"])
            with tab1:
                try:
                    with st.spinner('Performing computations. Please wait ...'):
                   
                        topic_model, topics, probs, embeddings = bertopic_vis(extype)
                        time.sleep(.5)
                        st.toast('Visualize Topics', icon='🏃')
                        fig1 = Vis_Topics(extype)
                       
                        time.sleep(.5)
                        st.toast('Visualize Document', icon='🏃')
                        fig2 = Vis_Documents(extype)
                       
                        time.sleep(.5)
                        st.toast('Visualize Document Hierarchy', icon='🏃')
                        fig3 = Vis_Hierarchy(extype)
                       
                        time.sleep(.5)
                        st.toast('Visualize Topic Similarity', icon='🏃')
                        fig4 = Vis_Heatmap(extype)
                       
                        time.sleep(.5)
                        st.toast('Visualize Terms', icon='🏃')
                        fig5 = Vis_Barchart(extype)
                       
                        bertab1, bertab2, bertab3, bertab4, bertab5 = st.tabs(["Visualize Topics", "Visualize Terms", "Visualize Documents",
                                                                              "Visualize Document Hierarchy", "Visualize Topic Similarity"])
                        
                        with bertab1:
                            st.plotly_chart(fig1, use_container_width=True)
                        with bertab2:
                            st.plotly_chart(fig5, use_container_width=True)
                        with bertab3:
                            st.plotly_chart(fig2, use_container_width=True)
                        with bertab4:  
                            st.plotly_chart(fig3, use_container_width=True)
                        with bertab5:
                            st.plotly_chart(fig4, use_container_width=True)
                      
                        #===download results===#
                        results = topic_model.get_topic_info()
                        resultf = pd.DataFrame(results)
                        resultcsv = resultf.to_csv().encode("utf-8")
                        st.download_button(
                            label = "Download Results",
                            data=resultcsv,
                            file_name="results.csv",
                            mime="text\csv",
                            on_click="ignore",
                        )

                except ValueError as e:
                    st.error('🙇‍♂️ Please raise the number of topics and click submit')
              
                except NameError as e:
                    st.warning('🖱️ Please click Submit')
    
            with tab2:
                st.markdown('**Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.** https://doi.org/10.48550/arXiv.2203.05794')
              
            with tab3:
                st.markdown('**Jeet Rawat, A., Ghildiyal, S., & Dixit, A. K. (2022, December 1). Topic modelling of legal documents using NLP and bidirectional encoder representations from transformers. Indonesian Journal of Electrical Engineering and Computer Science, 28(3), 1749.** https://doi.org/10.11591/ijeecs.v28.i3.pp1749-1755')
                st.markdown('**Yao, L. F., Ferawati, K., Liew, K., Wakamiya, S., & Aramaki, E. (2023, April 20). Disruptions in the Cystic Fibrosis Community’s Experiences and Concerns During the COVID-19 Pandemic: Topic Modeling and Time Series Analysis of Reddit Comments. Journal of Medical Internet Research, 25, e45249.** https://doi.org/10.2196/45249')

            with tab4:
                st.divider()
                st.subheader(':blue[BERTopic]', anchor=False)
                st.text("Click the camera icon on the top right menu")
                st.markdown("![Downloading visualization](https://raw.githubusercontent.com/faizhalas/library-tools/main/images/download_bertopic.jpg)")
                st.divider()
                st.subheader(':blue[Downloading CSV Results]', anchor=False)
                st.button("Download Results", on_click="ignore")
                st.text("Click Download results button at bottom of page")

    except Exception as e:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
