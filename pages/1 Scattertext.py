import streamlit as st
import scattertext as stx
import pandas as pd
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import time
import sys

#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.header("Scattertext")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def reset_all():
     st.cache_data.clear()

@st.cache_data(ttl=3600)
def get_ext(extype):
    extype = uploaded_file.name
    return extype

st.subheader('Put your file here...')

#===upload file===
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
            'AB': 'Abstract',
            'PY': 'Year'}
    papers = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
    papers.rename(columns=col_dict, inplace=True)
    return papers

@st.cache_data(ttl=3600)
def get_data(extype): 
    df_col = sorted(papers.select_dtypes(include=['object']).columns.tolist())
    list_title = [col for col in df_col if col.lower() == "title"]
    abstract_pattern = re.compile(r'abstract', re.IGNORECASE)
    list_abstract = [col for col in df_col if abstract_pattern.search(col)]

    if all(col in df_col for col in list_title) and all(col in df_col for col in list_abstract):
        selected_cols = list_title + list_abstract
    elif all(col in df_col for col in list_title):
        selected_cols = list_title
    elif all(col in df_col for col in list_abstract):
        selected_cols = list_abstract
    else:
        selected_cols = df_col

    if not selected_cols:
        selected_cols = df_col
    
    return df_col, selected_cols

@st.cache_data(ttl=3600)
def check_comparison(extype):
    comparison = ['Word-to-word', 'Manual label']
    
    if any('year' in col.lower() for col in papers.columns):
        comparison.append('Years')
    if any('source title' in col.lower() for col in papers.columns):
        comparison.append('Sources')

    comparison.sort(reverse=False)
    return comparison

#===clean csv===
@st.cache_data(ttl=3600, show_spinner=False)
def clean_csv(extype):
    paper = papers.dropna(subset=[ColCho])
                 
    #===mapping===
    paper[ColCho].map(lambda x: x.lower())
    if rem_punc:
        paper[ColCho] = paper[ColCho].map(lambda x: re.sub('[,:;\.!-?•=]', ' ', x))
        paper[ColCho] = paper[ColCho].str.replace('\u201c|\u201d', '', regex=True) 
    if rem_copyright:
        paper[ColCho] = paper[ColCho].map(lambda x: re.sub('©.*', '', x))
        
    #===stopword removal===
    stop = stopwords.words('english')
    paper[ColCho].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
          
    #===lemmatize===
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    paper[ColCho].apply(lemmatize_words)
    
    words_rmv = [word.strip() for word in words_to_remove.split(";")]
    remove_set = set(words_rmv)
    def remove_words(text):
        words = text.split()  
        cleaned_words = [word for word in words if word not in remove_set]
        return ' '.join(cleaned_words) 
    paper[ColCho] = paper[ColCho].apply(remove_words)
         
    return paper

@st.cache_data(ttl=3600)
def get_minmax(extype):
    MIN = int(papers['Year'].min())
    MAX = int(papers['Year'].max())
    GAP = MAX - MIN
    MID = round((MIN + MAX) / 2)
    return MIN, MAX, GAP, MID

@st.cache_data(ttl=3600)
def running_scattertext(cat_col, catname, noncatname):
    try:
        corpus = stx.CorpusFromPandas(filtered_df,         
                                category_col = cat_col,
                                text_col = ColCho,
                                nlp = stx.whitespace_nlp_with_sentences,
                                ).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count = min_term)        
                                
        st.toast('Building corpus completed', icon='🎉')
            
        try:
            html = stx.produce_scattertext_explorer(corpus,
                                                category = catname,
                                                category_name = catname,
                                                not_category_name = noncatname,
                                                width_in_pixels = 900,
                                                minimum_term_frequency = 0,
                                                metadata = filtered_df['Title'])
    
        except KeyError:
            html = stx.produce_scattertext_explorer(corpus,
                                                category = catname,
                                                category_name = catname,
                                                not_category_name = noncatname,
                                                width_in_pixels = 900,
                                                minimum_term_frequency = 0,
                                                save_svg_button=True)
    
        st.toast('Process completed', icon='🎉')
        time.sleep(1)
        st.toast('Visualizing', icon='⏳')
        st.components.v1.html(html, height = 1200, scrolling = True) 

    except ValueError:
        st.warning('Please increase the Minimum term count in the advanced settings.', icon="⚠️")
        sys.exit()

@st.cache_data(ttl=3600)
def df_w2w(search_terms1, search_terms2):
    selected_col = [ColCho]
    dfs1 = pd.DataFrame()
    for term in search_terms1:
        dfs1 = pd.concat([dfs1, paper[paper[selected_col[0]].str.contains(r'\b' + term + r'\b', case=False, na=False)]], ignore_index=True)
    dfs1['Topic'] = 'First Term'
        
    dfs2 = pd.DataFrame()
    for term in search_terms2:
        dfs2 = pd.concat([dfs2, paper[paper[selected_col[0]].str.contains(r'\b' + term + r'\b', case=False, na=False)]], ignore_index=True)
    dfs2['Topic'] = 'Second Term'
    filtered_df = pd.concat([dfs1, dfs2], ignore_index=True)
    
    return dfs1, dfs2, filtered_df

@st.cache_data(ttl=3600)
def df_sources(stitle1, stitle2):
    dfs1 = paper[paper['Source title'].str.contains(stitle1, case=False, na=False)]
    dfs1['Topic'] = stitle1
    dfs2 = paper[paper['Source title'].str.contains(stitle2, case=False, na=False)]
    dfs2['Topic'] = stitle2
    filtered_df = pd.concat([dfs1, dfs2], ignore_index=True)

    return filtered_df  

@st.cache_data(ttl=3600)
def df_years(first_range, second_range):
    first_range_filter_df = paper[(paper['Year'] >= first_range[0]) & (paper['Year'] <= first_range[1])].copy()
    first_range_filter_df['Topic Range'] = 'First range'
        
    second_range_filter_df = paper[(paper['Year'] >= second_range[0]) & (paper['Year'] <= second_range[1])].copy()
    second_range_filter_df['Topic Range'] = 'Second range'
        
    filtered_df = pd.concat([first_range_filter_df, second_range_filter_df], ignore_index=True)

    return filtered_df 

#===Read data===
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'], on_change=reset_all)

if uploaded_file is not None:
    extype = get_ext(uploaded_file)

    if extype.endswith('.csv'):
         papers = upload(extype) 
    elif extype.endswith('.txt'):
         papers = conv_txt(extype)

    df_col, selected_cols = get_data(extype)
    comparison = check_comparison(extype)

    #Menu
    c1, c2, c3 = st.columns([4,0.1,4])
    ColCho = c1.selectbox(
            'Choose column to analyze',
            (selected_cols), on_change=reset_all)

    c2.write('')

    compare = c3.selectbox(
            'Type of comparison',
            (comparison), on_change=reset_all)
    
    with st.expander("🧮 Show advance settings"):
        y1, y2 = st.columns([8,2])
        t1, t2 = st.columns([3,3])
        words_to_remove = y1.text_input('Input your text', on_change=reset_all, placeholder='Remove specific words. Separate words by semicolons (;)')
        min_term = y2.number_input("Minimum term count", min_value=0, max_value=10, value=3, step=1, on_change=reset_all)
        rem_copyright = t1.toggle('Remove copyright statement', value=True, on_change=reset_all)
        rem_punc = t2.toggle('Remove punctuation', value=False, on_change=reset_all)

    st.info('Scattertext is an expensive process when dealing with a large volume of text with our existing resources. Please kindly wait until the visualization appears.', icon="ℹ️")
    
    paper = clean_csv(extype)

    tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])

    with tab1:
         #===visualization===
        if compare == 'Word-to-word':
            col1, col2, col3 = st.columns([4,0.1,4])
            text1 = col1.text_input('First Term', on_change=reset_all, placeholder='put comma if you have more than one')
            search_terms1 = [term.strip() for term in text1.split(",") if term.strip()]
            col2.write('')
            text2 = col3.text_input('Second Term', on_change=reset_all, placeholder='put comma if you have more than one')
            search_terms2 = [term.strip() for term in text2.split(",") if term.strip()]
            
            dfs1, dfs2, filtered_df = df_w2w(search_terms1, search_terms2)
    
            if dfs1.empty and dfs2.empty:
                st.warning('We cannot find anything in your document.', icon="⚠️")
            elif dfs1.empty:
                st.warning(f'We cannot find {text1} in your document.', icon="⚠️")
            elif dfs2.empty:
                st.warning(f'We cannot find {text2} in your document.', icon="⚠️")
            else:
                with st.spinner('Processing. Please wait until the visualization comes up'):
                    running_scattertext('Topic', 'First Term', 'Second Term')
    
        elif compare == 'Manual label':
            col1, col2, col3 = st.columns(3)
    
            df_col_sel = sorted([col for col in paper.columns.tolist()])
                 
            column_selected = col1.selectbox(
                'Choose column',
                (df_col_sel), on_change=reset_all)
    
            list_words = paper[column_selected].values.tolist()
            list_unique = sorted(list(set(list_words)))
            
            if column_selected is not None:
                label1 = col2.selectbox(
                    'Choose first label',
                    (list_unique), on_change=reset_all)
    
                default_index = 0 if len(list_unique) == 1 else 1
                label2 = col3.selectbox(
                    'Choose second label',
                    (list_unique), on_change=reset_all, index=default_index)
    
            filtered_df = paper[paper[column_selected].isin([label1, label2])].reset_index(drop=True)
            
            with st.spinner('Processing. Please wait until the visualization comes up'):
                running_scattertext(column_selected, label1, label2)
    
        elif compare == 'Sources':
            col1, col2, col3 = st.columns([4,0.1,4])
    
            unique_stitle = set()
            unique_stitle.update(paper['Source title'].dropna())
            list_stitle = sorted(list(unique_stitle))
                 
            stitle1 = col1.selectbox(
                'Choose first label',
                (list_stitle), on_change=reset_all)
            col2.write('')
            default_index = 0 if len(list_stitle) == 1 else 1
            stitle2 = col3.selectbox(
                'Choose second label',
                (list_stitle), on_change=reset_all, index=default_index)
    
            filtered_df = df_sources(stitle1, stitle2)
    
            with st.spinner('Processing. Please wait until the visualization comes up'):
                running_scattertext('Source title', stitle1, stitle2)
    
        elif compare == 'Years':
            col1, col2, col3 = st.columns([4,0.1,4])
            
            MIN, MAX, GAP, MID = get_minmax(extype)
            if (GAP != 0):
                first_range = col1.slider('First Range', min_value=MIN, max_value=MAX, value=(MIN, MID), on_change=reset_all)
                col2.write('')
                second_range = col3.slider('Second Range', min_value=MIN, max_value=MAX, value=(MID, MAX), on_change=reset_all)
            
                filtered_df = df_years(first_range, second_range)
    
                with st.spinner('Processing. Please wait until the visualization comes up'):
                    running_scattertext('Topic Range', 'First range', 'Second range')

            else:
                st.write('You only have data in ', (MAX))

    with tab2:
        st.markdown('**Kessler, J.S. (2017). Scattertext: a Browser-Based Tool for Visualizing how Corpora Differ.** https://doi.org/10.48550/arXiv.1703.00565')

    with tab3:
        st.markdown('**Marrone, M., & Linnenluecke, M.K. (2020). Interdisciplinary Research Maps: A new technique for visualizing research topics. PLoS ONE, 15.** https://doi.org/10.1371/journal.pone.0242283')
        st.markdown('**Moreno, A., & Iglesias, C.A. (2021). Understanding Customers’ Transport Services with Topic Clustering and Sentiment Analysis. Applied Sciences.** https://doi.org/10.3390/app112110169')
        st.markdown('**Sánchez-Franco, M.J., & Rey-Tienda, S. (2023). The role of user-generated content in tourism decision-making: an exemplary study of Andalusia, Spain. Management Decision.** https://doi.org/10.1108/MD-06-2023-0966')
