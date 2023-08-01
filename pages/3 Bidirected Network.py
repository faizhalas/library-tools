#import module
import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from streamlit_agraph import agraph, Node, Edge, Config
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import sys

#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Biderected Keywords Network")
st.subheader('Put your file here...')

#===clear cache===
def reset_all():
     st.cache_data.clear()

#===check type===
@st.cache_data(ttl=3600)
def get_ext(extype):
    extype = uploaded_file.name
    return extype

@st.cache_data(ttl=3600)
def upload(extype):
    papers = pd.read_csv(uploaded_file)
    return papers

@st.cache_data(ttl=3600)
def conv_txt(extype):
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DT': 'Document Type',
            'DE': 'Author Keywords',
            'ID': 'Keywords Plus'}
    papers = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
    papers.rename(columns=col_dict, inplace=True)
    return papers

#===Read data===
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'], on_change=reset_all)

if uploaded_file is not None:
    extype = get_ext(uploaded_file)
    if extype.endswith('.csv'):
         papers = upload(extype) 
    elif extype.endswith('.txt'):
         papers = conv_txt(extype)
    
    @st.cache_data(ttl=3600)
    def get_data_arul(extype):
        list_of_column_key = list(papers.columns)
        list_of_column_key = [k for k in list_of_column_key if 'Keyword' in k]
        return papers, list_of_column_key
     
    papers, list_of_column_key = get_data_arul(extype)

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
             'Choose method',
           ('Stemming', 'Lemmatization'), on_change=reset_all)
    with col2:
        keyword = st.selectbox(
            'Choose column',
           (list_of_column_key), on_change=reset_all)


    #===body=== 
    @st.cache_data(ttl=3600)
    def clean_arul(extype):
        global keyword, papers
        try:
            arul = papers.dropna(subset=[keyword])
        except KeyError:
            st.error('Error: Please check your Author/Index Keywords column.')
            sys.exit(1)
        arul[keyword] = arul[keyword].map(lambda x: re.sub('-—–', ' ', x))
        arul[keyword] = arul[keyword].map(lambda x: re.sub('; ', ' ; ', x))
        arul[keyword] = arul[keyword].map(lambda x: x.lower())
        arul[keyword] = arul[keyword].dropna()
        return arul

    arul = clean_arul(extype)   

    #===stem/lem===
    @st.cache_data(ttl=3600)
    def lemma_arul(extype):
        lemmatizer = WordNetLemmatizer()
        def lemmatize_words(text):
             words = text.split()
             words = [lemmatizer.lemmatize(word) for word in words]
             return ' '.join(words)
        arul[keyword] = arul[keyword].apply(lemmatize_words)
        return arul
    
    @st.cache_data(ttl=3600)
    def stem_arul(extype):
        stemmer = SnowballStemmer("english")
        def stem_words(text):
            words = text.split()
            words = [stemmer.stem(word) for word in words]
            return ' '.join(words)
        arul[keyword] = arul[keyword].apply(stem_words)
        return arul

    if method is 'Lemmatization':
        arul = lemma_arul(extype)
    else:
        arul = stem_arul(extype)
    
    @st.cache_data(ttl=3600)
    def arm(extype):
        arule = arul[keyword].str.split(' ; ')
        arule_list = arule.values.tolist()  
        te_ary = te.fit(arule_list).transform(arule_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        return df
    df = arm(extype)

    col1, col2, col3 = st.columns(3)
    with col1:
        supp = st.slider(
            'Select value of Support',
            0.001, 1.000, (0.010), on_change=reset_all)
    with col2:
        conf = st.slider(
            'Select value of Confidence',
            0.001, 1.000, (0.050), on_change=reset_all)
    with col3:
        maxlen = st.slider(
            'Maximum length of the itemsets generated',
            2, 8, (2), on_change=reset_all)

    tab1, tab2, tab3 = st.tabs(["📈 Result & Generate visualization", "📃 Reference", "📓 Recommended Reading"])
    
    with tab1:
        #===Association rules===
        @st.cache_data(ttl=3600)
        def freqitem(extype):
            freq_item = fpgrowth(df, min_support=supp, use_colnames=True, max_len=maxlen)
            return freq_item
        
        @st.cache_data(ttl=3600)
        def arm_table(extype):
            res = association_rules(freq_item, metric='confidence', min_threshold=conf) 
            res = res[['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'conviction']]
            res['antecedents'] = res['antecedents'].apply(lambda x: ', '.join(list(x))).astype('unicode')
            res['consequents'] = res['consequents'].apply(lambda x: ', '.join(list(x))).astype('unicode')
            restab = res
            return res, restab

        freq_item = freqitem(extype)
        st.write('🚨 The more data you have, the longer you will have to wait.')

        if freq_item.empty:
            st.error('Please lower your value.', icon="🚨")
        else:
            res, restab = arm_table(extype)
            st.dataframe(restab, use_container_width=True)
                   
             #===visualize===
                
            if st.button('📈 Generate network visualization', on_click=reset_all):
                with st.spinner('Visualizing, please wait ....'): 
                     @st.cache_data(ttl=3600)
                     def map_node(extype):
                        res['to'] = res['antecedents'] + ' → ' + res['consequents'] + '\n Support = ' +  res['support'].astype(str) + '\n Confidence = ' +  res['confidence'].astype(str) + '\n Conviction = ' +  res['conviction'].astype(str)
                        res_ant = res[['antecedents','antecedent support']].rename(columns={'antecedents': 'node', 'antecedent support': 'size'}) #[['antecedents','antecedent support']]
                        res_con = res[['consequents','consequent support']].rename(columns={'consequents': 'node', 'consequent support': 'size'}) #[['consequents','consequent support']]
                        res_node = pd.concat([res_ant, res_con]).drop_duplicates(keep='first')
                        return res_node, res
                     
                     res_node, res = map_node(extype)

                     @st.cache_data(ttl=3600)
                     def arul_network(extype):
                        nodes = []
                        edges = []

                        for w,x in zip(res_node['size'], res_node['node']):
                            nodes.append( Node(id=x, 
                                            label=x,
                                            size=50*w+10,
                                            shape="circularImage",
                                            labelHighlightBold=True,
                                            group=x,
                                            opacity=10,
                                            mass=1,
                                            image="https://upload.wikimedia.org/wikipedia/commons/f/f1/Eo_circle_yellow_circle.svg") 
                                    )   

                        for y,z,a,b in zip(res['antecedents'],res['consequents'],res['confidence'],res['to']):
                            edges.append( Edge(source=y, 
                                            target=z,
                                            title=b,
                                            width=a*2,
                                            physics=True,
                                            smooth=True
                                            ) 
                                    )  
                        return nodes, edges

                     nodes, edges = arul_network(extype)
                     config = Config(width=1200,
                                     height=800,
                                     directed=True, 
                                     physics=True, 
                                     hierarchical=False,
                                     maxVelocity=5
                                     )

                     return_value = agraph(nodes=nodes, 
                                           edges=edges, 
                                           config=config)
    with tab2:
         st.markdown('**Santosa, F. A. (2023). Adding Perspective to the Bibliometric Mapping Using Bidirected Graph. Open Information Science, 7(1), 20220152.** https://doi.org/10.1515/opis-2022-0152')
         
    with tab2:
        st.markdown('**Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. In ACM SIGMOD Record (Vol. 22, Issue 2, pp. 207–216). Association for Computing Machinery (ACM).** https://doi.org/10.1145/170036.170072')
        st.markdown('**Brin, S., Motwani, R., Ullman, J. D., & Tsur, S. (1997). Dynamic itemset counting and implication rules for market basket data. ACM SIGMOD Record, 26(2), 255–264.** https://doi.org/10.1145/253262.253325')
        st.markdown('**Edmonds, J., & Johnson, E. L. (2003). Matching: A Well-Solved Class of Integer Linear Programs. Combinatorial Optimization — Eureka, You Shrink!, 27–30.** https://doi.org/10.1007/3-540-36478-1_3') 
        st.markdown('**Li, M. (2016, August 23). An exploration to visualise the emerging trends of technology foresight based on an improved technique of co-word analysis and relevant literature data of WOS. Technology Analysis & Strategic Management, 29(6), 655–671.** https://doi.org/10.1080/09537325.2016.1220518')
