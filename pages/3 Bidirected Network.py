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
import time
import json
from tools import sourceformat as sf
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
import altair_nx as anx

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
    st.page_link("pages/9 WordCloud.py", label = "WordCloud", icon = "9️⃣")

with st.expander("Before you start", expanded = True):

        tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
        with tab1:
            st.write("The use of network text analysis by librarians can be quite beneficial. Finding hidden correlations and connections in textual material is a significant advantage. Using network text analysis tools, librarians can improve knowledge discovery, obtain deeper insights, and support scholars meaningfully, ultimately enhancing the library's services and resources. This menu provides a two-way relationship instead of the general network of relationships to enhance the co-word analysis. Since it is based on ARM, you may obtain transactional data information using this menu. Please name the column in your file 'Keyword' instead.")
            st.divider()
            st.write('💡 The idea came from this:') 
            st.write('Santosa, F. A. (2023). Adding Perspective to the Bibliometric Mapping Using Bidirected Graph. Open Information Science, 7(1), 20220152. https://doi.org/10.1515/opis-2022-0152')

        with tab2:
            st.text("1. Put your file.")
            st.text("2. Choose your preferable method. Picture below may help you to choose wisely.")
            st.markdown("![Source: https://studymachinelearning.com/stemming-and-lemmatization/](https://studymachinelearning.com/wp-content/uploads/2019/09/stemmin_lemm_ex-1.png)")
            st.text('Source: https://studymachinelearning.com/stemming-and-lemmatization/')
            st.text("3. Choose the value of Support and Confidence. If you're not sure how to use it please read the article above or just try it!")
            st.text("4. You can see the table and a simple visualization before making a network visualization.")
            st.text('5. Click "Generate network visualization" to see the network')
            st.error("The more data on your table, the more you'll see on network.", icon="🚨")
            st.error("If the table contains many rows, the network will take more time to process. Please use it efficiently.", icon="⌛")
            
        with tab3:
            st.code("""
            +----------------+------------------------+---------------------------------+
            |     Source     |       File Type        |             Column              |
            +----------------+------------------------+---------------------------------+
            | Scopus         | Comma-separated values | Author Keywords                 |
            |                | (.csv)                 | Index Keywords                  |
            +----------------+------------------------+---------------------------------+
            | Web of Science | Tab delimited file     | Author Keywords                 |
            |                | (.txt)                 | Keywords Plus                   |
            +----------------+------------------------+---------------------------------+
            | Lens.org       | Comma-separated values | Keywords (Scholarly Works)      |
            |                | (.csv)                 |                                 |
            +----------------+------------------------+---------------------------------+
            | Dimensions     | Comma-separated values | MeSH terms                      |
            |                | (.csv)                 |                                 |
            +----------------+------------------------+---------------------------------+
            | OpenAlex       | Comma-separated values | Keywords                        |
            |                | (.csv)                 |                                 |
            +----------------+------------------------+---------------------------------+
            | Other          | .csv .xls .xlsx        | Change your column to 'Keyword' |
            |                |                        | and separate the words with ';' |
            +----------------+------------------------+---------------------------------+
            | Hathitrust     | .json                  | htid (Hathitrust ID)            |
            +----------------+------------------------+---------------------------------+
            """, language=None)    

        with tab4:
            st.subheader(":blue[Download visualization]", anchor=False)
            st.text("Zoom in, zoom out, or shift the nodes as desired, then right-click and select Save image as ...")
            st.markdown("![Downloading graph](https://raw.githubusercontent.com/faizhalas/library-tools/main/images/download_bidirected.jpg)")     
            st.subheader(":blue[Download table as CSV]", anchor=False)
            st.text("Hover cursor over table, and click download arrow")
            st.markdown("![Downloading table](https://raw.githubusercontent.com/faizhalas/library-tools/refs/heads/main/images/tablenetwork.png)")
            
st.header("Bidirected Network", anchor=False)
st.subheader('Put your file here...', anchor=False)

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

    if "About the data" in papers.columns[0]:
        papers = sf.dim(papers)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        papers.rename(columns=col_dict, inplace=True)

    elif "ids.openalex" in papers.columns:
        papers.rename(columns={'keywords.display_name': 'Keywords'}, inplace=True)
        papers["Keywords"] = papers["Keywords"].astype(str).str.replace("|", "; ")
    
    return papers

@st.cache_data(ttl=3600)
def conv_txt(extype):
    if("PMID" in (uploaded_file.read()).decode()):
        uploaded_file.seek(0)
        papers = sf.medline(uploaded_file)
        print(papers)
        return papers
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DE': 'Author Keywords',
            'DT': 'Document Type',
            'AB': 'Abstract',
            'TC': 'Cited by',
            'PY': 'Year',
            'ID': 'Keywords Plus',
            'rights_date_used': 'Year'}
    uploaded_file.seek(0)
    papers = pd.read_csv(uploaded_file, sep='\t')
    
    if("htid" in papers.columns):
        papers = sf.htrc(papers)
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

@st.cache_data(ttl=3600)
def readxls(file):
    papers = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
    if "About the data" in papers.columns[0]:
        papers = sf.dim(papers)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        papers.rename(columns=col_dict, inplace=True)
    
    return papers

#===Read data===
uploaded_file = st.file_uploader('', type=['csv', 'txt', 'json', 'tar.gz', 'xml', 'xls', 'xlsx'], on_change=reset_all)

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
        elif extype.endswith(('.xls', '.xlsx')):
            papers = readxls(uploaded_file)
        
        @st.cache_data(ttl=3600)
        def get_data_arul(extype):
            list_of_column_key = list(papers.columns)
            list_of_column_key = [k for k in list_of_column_key if 'Keyword' in k]
            return papers, list_of_column_key
         
        papers, list_of_column_key = get_data_arul(extype)
    
        col1, col2 = st.columns(2)
        with col1:
            dispmethod = st.selectbox('Choose display method',
            ("Agraph", "Altair-nx"), on_change=reset_all)

            method = st.selectbox(
                 'Choose method',
               ('Lemmatization', 'Stemming'), on_change=reset_all)
        with col2:
            keyword = st.selectbox(
                'Choose column',
               (list_of_column_key), on_change=reset_all)
            if dispmethod=="Altair-nx":
                layout = st.selectbox(
                    'Choose graph layout',
                    ['Circular','Kamada Kawai','Random','Spring','Shell'])
    
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
                'Support',
                0.001, 1.000, (0.010), on_change=reset_all, help='Frequency of occurrence of keywords in a set of documents')
        with col2:
            conf = st.slider(
                'Confidence',
                0.001, 1.000, (0.050), on_change=reset_all, help='Presence of keywords in documents that included the antecedents')
        with col3:
            maxlen = st.slider(
                'Maximum length of the itemsets generated',
                2, 8, (2), on_change=reset_all, help='')
    
        tab1, tab2, tab3 = st.tabs(["📈 Result & Generate visualization", "📃 Reference", "📓 Recommended Reading"])
        
        with tab1:
            #===Association rules===
            @st.cache_data(ttl=3600)
            def freqitem(extype):
                freq_item = fpgrowth(df, min_support=supp, use_colnames=True, max_len=maxlen)
                return freq_item
    
            freq_item = freqitem(extype)
            col1, col2 = st.columns(2)
            with col1:
                 st.write('🚨 The more data you have, the longer you will have to wait.')
            with col2:
                 showall = st.checkbox('Show all nodes', value=True, on_change=reset_all)
    
            @st.cache_data(ttl=3600)
            def arm_table(extype):
                restab = association_rules(freq_item, metric='confidence', min_threshold=conf) 
                restab = restab[['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'conviction']]
                restab['antecedents'] = restab['antecedents'].apply(lambda x: ', '.join(list(x))).astype('unicode')
                restab['consequents'] = restab['consequents'].apply(lambda x: ', '.join(list(x))).astype('unicode')
                if showall:
                     restab['Show'] = True
                else:
                     restab['Show'] = False
                return restab 
             
            if freq_item.empty:
                st.error('Please lower your value.', icon="🚨")
            else:
                restab = arm_table(extype)
                restab = st.data_editor(restab, use_container_width=True, hide_index=True)
                res = restab[restab['Show'] == True] 
                       
                 #===visualize===
                    
                if st.button('📈 Generate network visualization', on_click=reset_all):
                    with st.spinner('Visualizing, please wait ....'): 
                        @st.cache_data(ttl=3600)
                        def map_node(extype):
                            res['to'] = res['antecedents'] + ' → ' + res['consequents'] + '\n Support = ' +  res['support'].astype(str) + '\n Confidence = ' +  res['confidence'].astype(str) + '\n Conviction = ' +  res['conviction'].astype(str)
                            res_ant = res[['antecedents','antecedent support']].rename(columns={'antecedents': 'node', 'antecedent support': 'size'}) 
                            res_con = res[['consequents','consequent support']].rename(columns={'consequents': 'node', 'consequent support': 'size'}) 
                            res_node = pd.concat([res_ant, res_con]).drop_duplicates(keep='first')
                            return res_node, res
                         
                        res_node, res = map_node(extype)

                        if dispmethod == "Agraph":

                            @st.cache_data(ttl=3600)
                            def arul_network(extype):
                                nodes = []
                                edges = []
        
                                for w,x in zip(res_node['size'], res_node['node']):
                                    nodes.append( Node(id=x, 
                                                    label=x,
                                                    size=50*w+10,
                                                    shape="dot",
                                                    labelHighlightBold=True,
                                                    group=x,
                                                    opacity=10,
                                                    mass=1)
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
                            time.sleep(1)
                            st.toast('Process completed', icon='📈')


                        elif(dispmethod=="Altair-nx"):
                            @st.cache_data(ttl=3600)
                            def graphmaker(__netgraph):

                                #add nodes, w is weight, x is node label
                                for w,x in zip(res_node['size'], res_node['node']):
                                    __netgraph.add_node(x, size = (400 + 2000*w))
                                #add edges, y is startpoint, z is endpoint, a is edge weight, b is title
                                for y,z,a,b in zip(res['antecedents'],res['consequents'],res['confidence'],res['to']):
                                    __netgraph.add_edge(y,z, weight = int(a*10))


                            #Make graph with NetworkX

                            G=nx.DiGraph()

                            graphmaker(G)

                            #Graph layout    
                            if(layout=="Spring"):
                                pos=nx.spring_layout(G)
                            elif(layout == "Kamada Kawai"):
                                pos=nx.kamada_kawai_layout(G)                    
                            elif(layout == "Circular"):
                                pos = nx.circular_layout(G)
                            elif(layout=="Random"):
                                pos = nx.random_layout(G)
                            elif(layout=="Shell"):
                                pos=nx.shell_layout(G)
                        
                            graph = anx.draw_networkx(G,pos, node_label = 'node',
                            edge_width = 'weight',
                            node_size = 'size',
                            curved_edges = True,
                            node_font_size=12,
                            edge_alpha = 0.25,
                            edge_colour = "grey",
                            node_colour = "royalblue",                    
                            chart_width=800,
                            chart_height=600).interactive()
                        
                            with st.container(border = True):
                                st.altair_chart(graph)
                        
        with tab2:
             st.markdown('**Santosa, F. A. (2023). Adding Perspective to the Bibliometric Mapping Using Bidirected Graph. Open Information Science, 7(1), 20220152.** https://doi.org/10.1515/opis-2022-0152')
             
        with tab3:
            st.markdown('**Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. In ACM SIGMOD Record (Vol. 22, Issue 2, pp. 207–216). Association for Computing Machinery (ACM).** https://doi.org/10.1145/170036.170072')
            st.markdown('**Brin, S., Motwani, R., Ullman, J. D., & Tsur, S. (1997). Dynamic itemset counting and implication rules for market basket data. ACM SIGMOD Record, 26(2), 255–264.** https://doi.org/10.1145/253262.253325')
            st.markdown('**Edmonds, J., & Johnson, E. L. (2003). Matching: A Well-Solved Class of Integer Linear Programs. Combinatorial Optimization — Eureka, You Shrink!, 27–30.** https://doi.org/10.1007/3-540-36478-1_3') 
            st.markdown('**Li, M. (2016, August 23). An exploration to visualise the emerging trends of technology foresight based on an improved technique of co-word analysis and relevant literature data of WOS. Technology Analysis & Strategic Management, 29(6), 655–671.** https://doi.org/10.1080/09537325.2016.1220518')
                  
    except:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
