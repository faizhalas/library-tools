import streamlit as st
import pandas as pd
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

st.page_link("https://www.coconut-libtool.com/the-app", label="Go to app", icon="🥥")

def reset_data():
     st.cache_data.clear()

#===check filetype===
@st.cache_data(ttl=3600)
def get_ext(extype):
    extype = uploaded_file.name
    return extype
     
#===upload===
@st.cache_data(ttl=3600)
def upload(extype):
    keywords = pd.read_csv(uploaded_file)
    if "dimensions" in uploaded_file.name.lower():
        keywords = sf.dim(keywords)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        keywords.rename(columns=col_dict, inplace=True)
    return keywords

@st.cache_data(ttl=3600)
def conv_txt(extype):
    if "pmc" in uploaded_file.name.lower():
        file = uploaded_file
        papers = sf.medline(file)
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
    'content_provider_code':'Source title'
    }
    keywords = pd.read_json(uploaded_file)
    keywords = sf.htrc(keywords)
    keywords['Cited by'] = keywords.groupby(['Keywords'])['Keywords'].transform('size')
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

st.header('File Checker', anchor=False)
st.subheader('Put your file here...', anchor=False)

#===read data===
uploaded_file = st.file_uploader('', type=['csv','txt','json', 'tar.gz', 'xml'], on_change=reset_data)

if uploaded_file is not None:
    extype = get_ext(uploaded_file)
    if extype.endswith('.csv'):
        data = upload(extype) 
                  
    elif extype.endswith('.txt'):
        data = conv_txt(extype)

    elif extype.endswith('.json'):
        data = conv_json(extype)
    
    elif extype.endswith('.tar.gz') or extype.endswith('.xml'):
        data = conv_pub(uploaded_file)


    col1, col2, col3 = st.columns(3)
  
    with col1:
        #===check keywords===  
        keycheck = list(data.columns)
        keycheck = [k for k in keycheck if 'Keyword' in k]
        container1 = st.container(border=True)
        
        if not keycheck:
            container1.subheader('❌ Keyword Stem', divider='red', anchor=False)
            container1.write("Unfortunately, you don't have a column containing keywords in your data. Please check again. If you want to use it in another column, please rename it to 'Keywords'.")
        else:
            container1.subheader('✔️ Keyword Stem', divider='blue', anchor=False)
            container1.write('Congratulations! You can use Keywords Stem')

        #===Visualization===
        if 'Publication Year' in data.columns:
                   data.rename(columns={'Publication Year': 'Year', 'Citing Works Count': 'Cited by',
                                         'Publication Type': 'Document Type', 'Source Title': 'Source title'}, inplace=True)
    
        col2check = ['Document Type','Source title','Cited by','Year']
        miss_col = [column for column in col2check if column not in data.columns]
        container2 = st.container(border=True)
        
        if not miss_col:
            container2.subheader('✔️ Sunburst', divider='blue', anchor=False)
            container2.write('Congratulations! You can use Sunburst')
        else:
            container2.subheader('❌ Sunburst', divider='red', anchor=False)
            miss_col_str = ', '.join(miss_col)
            container2.write(f"Unfortunately, you don't have: {miss_col_str}. Please check again.")           

    with col2:   
        #===check any obj===
        coldf = sorted(data.select_dtypes(include=['object']).columns.tolist())
        container3 = st.container(border=True)
                
        if not coldf or data.shape[0] < 2:
            container3.subheader('❌ Topic Modeling', divider='red', anchor=False)
            container3.write("Unfortunately, you don't have a column containing object in your data. Please check again.")
        else:
            container3.subheader('✔️ Topic Modeling', divider='blue', anchor=False)
            container3.write('Congratulations! You can use Topic Modeling')

        #===Burst===
        container4 = st.container(border=True)
        if not coldf or 'Year' not in data.columns:
            container4.subheader('❌ Burst Detection', divider='red', anchor=False)
            container4.write("Unfortunately, you don't have a column containing object in your data or a 'Year' column. Please check again.")
        else:
            container4.subheader('✔️ Burst Detection', divider='blue', anchor=False)
            container4.write('Congratulations! You can use Burst Detection')

    with col3:
        #===bidirected===    
        container5 = st.container(border=True)        
        if not keycheck:
            container5.subheader('❌ Bidirected Network', divider='red', anchor=False)
            container5.write("Unfortunately, you don't have a column containing keywords in your data. Please check again. If you want to use it in another column, please rename it to 'Keywords'.")
        else:
            container5.subheader('✔️ Bidirected Network', divider='blue', anchor=False)
            container5.write('Congratulations! You can use Bidirected Network')

        #===scattertext===
        container6 = st.container(border=True)   
        if not coldf or data.shape[0] < 2:
            container6.subheader('❌ Scattertext', divider='red', anchor=False)
            container6.write("Unfortunately, you don't have a column containing object in your data. Please check again.")
        else:
            container6.subheader('✔️ Scattertext', divider='blue', anchor=False)
            container6.write('Congratulations! You can use Scattertext')
