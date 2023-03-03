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
import plotly.express as px
from io import StringIO
from arulesviz import Arulesviz
from ipywidgets.embed import embed_minimal_html

st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)

#Read data

st.header("AR for Keywords")
   
#subhead
st.subheader('Put your CSV file and click generate')

uploaded_file = st.file_uploader("Choose a file")

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
         'Choose method',
       ('Stemming', 'Lemmatization'))
with col2:
    keyword = st.selectbox(
        'Choose column',
       ('Author Keywords', 'Index Keywords'))


 
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    papers = pd.read_csv(uploaded_file)
    arul = papers.dropna(subset=[keyword])
     
    arul[keyword] = arul[keyword].map(lambda x: re.sub('[(),:&\.!?•-]', '', x))
    arul[keyword] = arul[keyword].map(lambda x: re.sub(' ', '_', x))
    arul[keyword] = arul[keyword].map(lambda x: re.sub(';_', ' ', x))
    arul[keyword] = arul[keyword].map(lambda x: x.lower())
    arul = arul.apply(lambda row: nltk.word_tokenize(row[keyword]), axis=1)
    arul = arul.values.tolist()
    te_ary = te.fit(arul).transform(arul)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
col1, col2 = st.columns(2)
with col1:
    supp = st.slider(
        'Select value of Support',
        0.00, 1.00, (0.04))
with col2:
    conf = st.slider(
        'Select value of Confidence',
        0.00, 1.00, (0.07))


if st.button('📝 Show Table'):
    with st.spinner('Calculating, please wait ....'): 
        freq_item = fpgrowth(df, min_support=supp, use_colnames=True)
        res = association_rules(freq_item, metric='confidence', min_threshold=conf) 
        ras = res[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        st.write(ras)
       
                
if st.button('📈 Generate visualization'):
    with st.spinner('Visualizing, please wait ....'): 
        g = Arulesviz(arul, supp, conf, 0.5, products_to_drop=[])
        g.create_rules()
        fig = g.plot_graph(width=1500, directed=False, charge=-200, link_distance=30)

        ## Exports the ipywidget to HTML and embeds it to streamlit
        with StringIO() as f:
            embed_minimal_html(f, [fig], title="Visualization")
            fig_html = f.getvalue()
            st.components.v1.html(fig_html, width=1500, height=1000, scrolling=True)

#if st.button('📈 Generate visualization'):
#    with st.spinner('Visualizing, please wait ....'): 
#        res=pd.DataFrame()
#        fg = px.data.res()
#        fg = px.scatter_3d(res, x='confidence', y='support', z='lift',
#              color='lift')
#        st.plotly_chart(fg, use_container_width=False)
 
 
 
 
 
 
 
 
