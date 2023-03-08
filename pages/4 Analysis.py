#===import module===
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Visualization")
st.subheader('Put your CSV file and click generate')

#===body===
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None: 
    papers = pd.read_csv(uploaded_file)
    option = st.selectbox(
    'Please choose analysis',
    ('Sunburst', 'Etc'))
    
    #===sunburst===
    if option == 'Sunburst':
        MIN = int(papers['Year'].min())
        MAX = int(papers['Year'].max())
        YEAR = st.slider('Year', min_value=MIN, max_value=MAX, (2000, 2001))
        years = list(range(YEAR[0],YEAR[1]+1))
        papers = papers.loc[papers['Year'].isin(years)]
        if {'Document Type','Source title','Cited by','Year'}.issubset(papers.columns):
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
            st.plotly_chart(fig, height=800, width=1200) #use_container_width=True)
        else:
            st.error('We require this columns: Document Type, Source title, Cited by, Year', icon="🚨")
            
    #===else===        
    elif option == 'Etc':
        st.error('Under development', icon="🚨")
