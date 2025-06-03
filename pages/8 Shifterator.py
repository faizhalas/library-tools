import streamlit as st
import streamlit.components.v1 as components
import shifterator as sh
from shifterator import ProportionShift
import pandas as pd
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import time
import sys
import json
from tools import sourceformat as sf
from collections import Counter
import io

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
    
st.header("Shifterator", anchor=False)
st.subheader('Put your file here...', anchor=False)

def reset_all():
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def get_ext(extype):
    extype = uploaded_file.name
    return extype

#===upload file===
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

@st.cache_data(ttl=3600)
def get_data(extype): 
    df_col = sorted(papers.select_dtypes(include=['object']).columns.tolist())
    list_title = [col for col in df_col if col.lower() == "title"]
    abstract_pattern = re.compile(r'abstract', re.IGNORECASE)
    list_abstract = [col for col in df_col if abstract_pattern.search(col)]

    if all(col in df_col for col in list_title) and all(col in df_col for col in list_abstract):
        selected_cols = list_abstract + list_title
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
    paper[ColCho] = paper[ColCho].map(lambda x: x.lower())
    if rem_punc:
        paper[ColCho] = paper[ColCho].map(lambda x: re.sub('[,:;\.!-?•=]', ' ', x))
        paper[ColCho] = paper[ColCho].str.replace('\u201c|\u201d', '', regex=True) 
    if rem_copyright:
        paper[ColCho] = paper[ColCho].map(lambda x: re.sub('©.*', '', x))
        
    #===stopword removal===
    stop = stopwords.words('english')
    paper[ColCho] = paper[ColCho].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
          
    #===lemmatize===
    lemmatizer = WordNetLemmatizer()
    
    @st.cache_data(ttl=3600)
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
        
    paper[ColCho] = paper[ColCho].apply(lemmatize_words)
    
    words_rmv = [word.strip() for word in words_to_remove.split(";")]
    remove_set = set(words_rmv)
    
    @st.cache_data(ttl=3600)
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
def running_shifterator(dict1, dict2):
    try:
        if method_shifts == 'Proportion Shifts':
            proportion_shift = sh.ProportionShift(type2freq_1=dict1, type2freq_2=dict2)
            ax = proportion_shift.get_shift_graph(system_names = ['Topic 1', 'Topic 2'], title='Proportion Shifts')     
            
        elif method_shifts == 'Shannon Entropy Shifts':
            entropy_shift = sh.EntropyShift(type2freq_1=dict1,
                                type2freq_2=dict2,
                                base=2)
            ax = entropy_shift.get_shift_graph(system_names = ['Topic 1', 'Topic 2'], title='Shannon Entropy Shifts')     
            
        elif method_shifts == 'Tsallis Entropy Shifts':
            entropy_shift = sh.EntropyShift(type2freq_1=dict1,
                                type2freq_2=dict2,
                                base=2,
                                alpha=0.8)
            ax = entropy_shift.get_shift_graph(system_names = ['Topic 1', 'Topic 2'], title='Tsallis Entropy Shifts')     
            
        elif method_shifts == 'Kullback-Leibler Divergence Shifts':
            kld_shift = sh.KLDivergenceShift(type2freq_1=dict1,
                                 type2freq_2=dict2,
                                 base=2)
            ax = kld_shift.get_shift_graph(system_names = ['Topic 1', 'Topic 2'], title='Kullback-Leibler Divergence Shifts')     
            
        elif method_shifts == 'Jensen-Shannon Divergence Shifts':
            jsd_shift = sh.JSDivergenceShift(type2freq_1=dict1,
                                 type2freq_2=dict2,
                                 weight_1=0.5,
                                 weight_2=0.5,
                                 base=2,
                                 alpha=1)
            ax = jsd_shift.get_shift_graph(system_names = ['Topic 1', 'Topic 2'], title='Jensen-Shannon Divergence Shifts')     

        fig = ax.get_figure()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        
        return fig, buf

    except ValueError:
        st.warning('Please check your data.', icon="⚠️")
        sys.exit()

@st.cache_data(ttl=3600)
def df2dict(df_1, df_2):
    text1 = ' '.join(df_1.dropna().astype(str))
    text2 = ' '.join(df_2.dropna().astype(str))
                
    text1_clean = re.sub(r'\d+', '', text1)
    text2_clean = re.sub(r'\d+', '', text2)
                
    tokens1 = re.findall(r'\b\w+\b', text1_clean.lower())
    tokens2 = re.findall(r'\b\w+\b', text2_clean.lower())
                
    type2freq_1 = {k: int(v) for k, v in Counter(tokens1).items()}
    type2freq_2 = {k: int(v) for k, v in Counter(tokens2).items()}

    return type2freq_1, type2freq_2

@st.cache_data(ttl=3600)
def dict_w2w(search_terms1, search_terms2):
    selected_col = [ColCho]
    dfs1 = pd.DataFrame()
    for term in search_terms1:
        dfs1 = pd.concat([dfs1, paper[paper[selected_col[0]].str.contains(r'\b' + term + r'\b', case=False, na=False)]], ignore_index=True)
    dfs1['Topic'] = 'First Term'
    dfs1 = dfs1.drop_duplicates()
        
    dfs2 = pd.DataFrame()
    for term in search_terms2:
        dfs2 = pd.concat([dfs2, paper[paper[selected_col[0]].str.contains(r'\b' + term + r'\b', case=False, na=False)]], ignore_index=True)
    dfs2['Topic'] = 'Second Term'
    dfs2 = dfs2.drop_duplicates()
    
    type2freq_1, type2freq_2 = df2dict(dfs1[selected_col[0]], dfs2[selected_col[0]])
    
    return type2freq_1, type2freq_2

@st.cache_data(ttl=3600)
def dict_sources(stitle1, stitle2):
    selected_col = [ColCho]
    dfs1 = paper[paper['Source title'].str.contains(stitle1, case=False, na=False)]
    dfs1['Topic'] = stitle1
    dfs2 = paper[paper['Source title'].str.contains(stitle2, case=False, na=False)]
    dfs2['Topic'] = stitle2

    type2freq_1, type2freq_2 = df2dict(dfs1[selected_col[0]], dfs2[selected_col[0]])
    
    return type2freq_1, type2freq_2

@st.cache_data(ttl=3600)
def dict_years(first_range, second_range):
    selected_col = [ColCho]
    first_filter_df = paper[(paper['Year'] >= first_range[0]) & (paper['Year'] <= first_range[1])].copy()
    first_filter_df['Topic Range'] = 'First range'
        
    second_filter_df = paper[(paper['Year'] >= second_range[0]) & (paper['Year'] <= second_range[1])].copy()
    second_filter_df['Topic Range'] = 'Second range'

    type2freq_1, type2freq_2 = df2dict(first_filter_df[selected_col[0]], second_filter_df[selected_col[0]])
    
    return type2freq_1, type2freq_2
    

#===Read data===
uploaded_file = st.file_uploader('', type=['csv', 'txt', 'json', 'tar.gz','xml'], on_change=reset_all)

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
            y1, y2, y3 = st.columns([4,0.1,4])
            t1, t2 = st.columns([3,3])
            words_to_remove = y1.text_input('Input your text', on_change=reset_all, placeholder='Remove specific words. Separate words by semicolons (;)')
            method_shifts = y3.selectbox("Choose preferred method",('Proportion Shifts','Shannon Entropy Shifts', 'Tsallis Entropy Shifts','Kullback-Leibler Divergence Shifts', 
                                                                   'Jensen-Shannon Divergence Shifts'), on_change=reset_all)
            rem_copyright = t1.toggle('Remove copyright statement', value=True, on_change=reset_all)
            rem_punc = t2.toggle('Remove punctuation', value=False, on_change=reset_all)
    
        if method_shifts == 'Kullback-Leibler Divergence Shifts':
            st.info('The Kullback-Leibler Divergence is only well-defined if every single word in the comparison text is also in the reference text.', icon="ℹ️")
        
        paper = clean_csv(extype)
    
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help"])
    
        with tab1:
             #===visualization===
            if compare == 'Word-to-word':
                col1, col2, col3 = st.columns([4,0.1,4])
                text1 = col1.text_input('First Term', on_change=reset_all, placeholder='put comma if you have more than one')
                search_terms1 = [term.strip() for term in text1.split(",") if term.strip()]
                col2.write('')
                text2 = col3.text_input('Second Term', on_change=reset_all, placeholder='put comma if you have more than one')
                search_terms2 = [term.strip() for term in text2.split(",") if term.strip()]
                
                type2freq_1, type2freq_2 = dict_w2w(search_terms1, search_terms2)
        
                if not type2freq_1 and not type2freq_2:
                    st.warning('We cannot find anything in your document.', icon="⚠️")
                elif not type2freq_1:
                    st.warning(f'We cannot find {text1} in your document.', icon="⚠️")
                elif not type2freq_2:
                    st.warning(f'We cannot find {text2} in your document.', icon="⚠️")
                else:
                    with st.spinner('Processing. Please wait until the visualization comes up'):
                        fig, buf = running_shifterator(type2freq_1, type2freq_2)
                        st.pyplot(fig)
        
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
                
                dfs1 = filtered_df[filtered_df[column_selected] == label1].reset_index(drop=True)
                dfs2 = filtered_df[filtered_df[column_selected] == label2].reset_index(drop=True)

                type2freq_1, type2freq_2 = df2dict(dfs1[ColCho], dfs2[ColCho])
                
                with st.spinner('Processing. Please wait until the visualization comes up'):
                    fig, buf = running_shifterator(type2freq_1, type2freq_2)
                    st.pyplot(fig)
        
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
        
                type2freq_1, type2freq_2 = dict_sources(stitle1, stitle2)
        
                with st.spinner('Processing. Please wait until the visualization comes up'):
                    fig, buf = running_shifterator(type2freq_1, type2freq_2)
                    st.pyplot(fig)
        
            elif compare == 'Years':
                col1, col2, col3 = st.columns([4,0.1,4])
                
                MIN, MAX, GAP, MID = get_minmax(extype)
                if (GAP != 0):
                    first_range = col1.slider('First Range', min_value=MIN, max_value=MAX, value=(MIN, MID), on_change=reset_all)
                    col2.write('')
                    second_range = col3.slider('Second Range', min_value=MIN, max_value=MAX, value=(MID, MAX), on_change=reset_all)
                
                    type2freq_1, type2freq_2 = dict_years(first_range, second_range)
        
                    with st.spinner('Processing. Please wait until the visualization comes up'):
                        fig, buf = running_shifterator(type2freq_1, type2freq_2)
                        st.pyplot(fig)

                else:
                    st.write('You only have data in ', (MAX))

            d1, d2 = st.columns(2)
                
            d1.download_button(
                label="📥 Download Graph",
                data=buf,
                file_name="shifterator.png",
                mime="image/png"
            )

            @st.cache_data(ttl=3600)
            def shifts_dfs(type2freq_1, type2freq_2):
                proportion_shift = ProportionShift(type2freq_1=type2freq_1, type2freq_2=type2freq_2)
                
                words = list(proportion_shift.types)
                shift_scores = proportion_shift.get_shift_scores()
                freq1 = proportion_shift.type2freq_1
                freq2 = proportion_shift.type2freq_2

                data = []
                for word, score in shift_scores.items():
                    data.append({
                        'word': word,
                        'freq_text1': proportion_shift.type2freq_1.get(word, 0),
                        'freq_text2': proportion_shift.type2freq_2.get(word, 0),
                        'shift_score': score
                    })
                
                df_shift = pd.DataFrame(data)
                df_shift = df_shift.sort_values('shift_score')
                
                return df_shift.to_csv(index=False).encode('utf-8')

            csv = shifts_dfs(type2freq_1, type2freq_2)

            d2.download_button(
                "📥 Click to download result",
                csv,
                "shiftertor_dataframe.csv",
                "text/csv")
    
        with tab2:
            st.markdown('**Gallagher, R.J., Frank, M.R., Mitchell, L. et al. (2021). Generalized Word Shift Graphs: A Method for Visualizing and Explaining Pairwise Comparisons Between Texts. EPJ Data Science, 10(4).** https://doi.org/10.1140/epjds/s13688-021-00260-3')
    
        with tab3:
            st.markdown('**Sánchez-Franco, M. J., & Rey-Tienda, S. (2023). The role of user-generated content in tourism decision-making: an exemplary study of Andalusia, Spain. Management Decision, 62(7).** https://doi.org/10.1108/md-06-2023-0966')
            st.markdown('**Ipek Baris Schlicht, Fernandez, E., Chulvi, B., & Rosso, P. (2023). Automatic detection of health misinformation: a systematic review. Journal of Ambient Intelligence and Humanized Computing, 15.** https://doi.org/10.1007/s12652-023-04619-4')
            st.markdown('**Torricelli, M., Falkenberg, M., Galeazzi, A., Zollo, F., Quattrociocchi, W., & Baronchelli, A. (2023). Hurricanes Increase Climate Change Conversations on Twitter. PLOS Climate, 2(11)** https://doi.org/10.1371/journal.pclm.0000277')

        with tab4:
            st.subheader(':blue[Result]', anchor=False)
            st.button('📥 Download Graph')
            st.text("Click Download Graph button.")  

            st.divider()
            st.subheader(':blue[Shifterator Dataframe]', anchor=False)
            st.button('📥 Click to download result')
            st.text("Click the Download button to get the CSV result.") 

    except Exception as e:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
