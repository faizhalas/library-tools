#===import module===
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
    return papers

@st.cache_data(ttl=3600)
def conv_txt(extype):
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DT': 'Document Type',
            'DE': 'Author Keywords',
            'ID': 'Keywords Plus',
            'AB': 'Abstract',
            'TC': 'Cited by',
            'PY': 'Year',}
    papers = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
    papers.rename(columns=col_dict, inplace=True)
    return papers

#===Read data===
uploaded_file = st.file_uploader('', type=['csv', 'txt'], on_change=reset_all)

if uploaded_file is not None:
    try:
        extype = get_ext(uploaded_file)
        if extype.endswith('.csv'):
             papers = upload(extype) 
       
        elif extype.endswith('.txt'):
             papers = conv_txt(extype)
        
        @st.cache_data(ttl=3600)
        def get_minmax(extype):
            extype = extype
            MIN = int(papers['Year'].min())
            MAX = int(papers['Year'].max())
            GAP = MAX - MIN
            return papers, MIN, MAX, GAP
        
        tab1, tab2 = st.tabs(["📈 Generate visualization", "📓 Recommended Reading"])
        
        with tab1:    
            #===sunburst===
            try:
                papers, MIN, MAX, GAP = get_minmax(extype)
            except KeyError:
                st.error('Error: Please check again your columns.')
                sys.exit(1)
            
            if (GAP != 0):
                YEAR = st.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX), on_change=reset_all)
            else:
                st.write('You only have data in ', (MAX))
                YEAR = (MIN, MAX)
            
            @st.cache_data(ttl=3600)
            def listyear(extype):
                global papers
                years = list(range(YEAR[0],YEAR[1]+1))
                papers = papers.loc[papers['Year'].isin(years)]
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
    
    except:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
