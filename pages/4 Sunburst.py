#===import module===
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sys
from tools import sourceformat as sf


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
    
st.header("Sunburst Visualization", anchor=False)
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
    #lens.org
    if 'Publication Year' in papers.columns:
            papers.rename(columns={'Publication Year': 'Year', 'Citing Works Count': 'Cited by',
                                    'Publication Type': 'Document Type', 'Source Title': 'Source title'}, inplace=True)
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
    if "pmc" in uploaded_file.name.lower():
        file = uploaded_file
        papers = sf.medline(file)
    else:
        col_dict = {'TI': 'Title',
                'SO': 'Source title',
                'DT': 'Document Type',
                'AB': 'Abstract',
                'PY': 'Year'}
        papers = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
        papers.rename(columns=col_dict, inplace=True)
    print(papers)
    return papers


@st.cache_data(ttl=3600)
def conv_json(extype):
    col_dict={'title': 'title',
    'rights_date_used': 'Year',
    'content_provider_code': 'Document Type',
    'Keywords':'Source title'
    }
    keywords = pd.read_json(uploaded_file)
    keywords = sf.htrc(keywords)
    keywords['Cited by'] = keywords.groupby(['Keywords'])['Keywords'].transform('size')
    keywords.rename(columns=col_dict,inplace=True)
    return keywords

def conv_pub(extype):
    if (get_ext(extype)).endswith('.tar.gz'):
        bytedata = extype.read()
        keywords = sf.readPub(bytedata)
    elif (get_ext(extype)).endswith('.xml'):
        bytedata = extype.read()
        keywords = sf.readxml(bytedata)
    keywords['Cited by'] = keywords.groupby(['Keywords'])['Keywords'].transform('size')
    st.write(keywords)
    return keywords

#===Read data===
uploaded_file = st.file_uploader('', type=['csv', 'txt','json','tar.gz', 'xml'], on_change=reset_all)

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

        @st.cache_data(ttl=3600)
        def get_minmax(extype):
            extype = extype
            MIN = int(papers['Year'].min())
            MAX = int(papers['Year'].max())
            MIN1 = int(papers['Cited by'].min())
            MAX1 = int(papers['Cited by'].max())
            GAP = MAX - MIN
            return papers, MIN, MAX, GAP, MIN1, MAX1

        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📓 Recommended Reading","Download help"])
        
        with tab1:    
            #===sunburst===
            try:
                papers, MIN, MAX, GAP, MIN1, MAX1 = get_minmax(extype)
            except KeyError:
                st.error('Error: Please check again your columns.')
                sys.exit(1)
            
            if (GAP != 0):
                YEAR = st.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX), on_change=reset_all)
                KEYLIM = st.slider('Cited By Count',min_value = MIN1, max_value = MAX1, value = (MIN1,MAX1), on_change=reset_all)
            else:
                st.write('You only have data in ', (MAX))
                YEAR = (MIN, MAX)
                KEYLIM = (MIN1,MAX1)
            @st.cache_data(ttl=3600)
            def listyear(extype):
                global papers
                years = list(range(YEAR[0],YEAR[1]+1))
                cited = list(range(KEYLIM[0],KEYLIM[1]+1))
                papers = papers.loc[papers['Year'].isin(years)]
                papers = papers.loc[papers['Cited by'].isin(cited)]
                return years, papers
            
            @st.cache_data(ttl=3600)
            def vis_sunbrust(extype):
                papers['Cited by'] = papers['Cited by'].fillna(0)
                vis = pd.DataFrame()
                vis[['doctype','source','citby','year']] = papers[['Document Type','Source title','Cited by','Year']]
                viz=vis.groupby(['doctype', 'source', 'year'])['citby'].agg(['sum','count']).reset_index()  
                viz.rename(columns={'sum': 'cited by', 'count': 'total docs'}, inplace=True)
                                
                fig = px.sunburst(viz, path=['doctype', 'source', 'year'], values='total docs',
                              color='cited by', 
                              color_continuous_scale='RdBu',
                              color_continuous_midpoint=np.average(viz['cited by'], weights=viz['total docs']))
                fig.update_layout(height=800, width=1200)
                return fig
            
            years, papers = listyear(extype)
    
            if {'Document Type','Source title','Cited by','Year'}.issubset(papers.columns):
                fig = vis_sunbrust(extype)
                st.plotly_chart(fig, height=800, width=1200) #use_container_width=True)
               
            else:
                st.error('We require these columns: Document Type, Source title, Cited by, Year', icon="🚨")
        
        with tab2:
            st.markdown('**numpy.average — NumPy v1.24 Manual. (n.d.). Numpy.Average — NumPy v1.24 Manual.** https://numpy.org/doc/stable/reference/generated/numpy.average.html')
            st.markdown('**Sunburst. (n.d.). Sunburst Charts in Python.** https://plotly.com/python/sunburst-charts/')

        with tab3:
            st.text("Click the camera icon on the top right menu (you may need to hover your cursor within the visualization)")
            st.markdown("![Downloading visualization](https://raw.githubusercontent.com/faizhalas/library-tools/main/images/download_bertopic.jpg)")
    except:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()

