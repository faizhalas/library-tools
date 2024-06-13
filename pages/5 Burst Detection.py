import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
from burst_detection import burst_detection, enumerate_bursts, burst_weights
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    st.page_link("Home.py", label="Home", icon="🏠")
    st.page_link("pages/1 Scattertext.py", label="Scattertext", icon="1️⃣")
    st.page_link("pages/2 Topic Modeling.py", label="Topic Modeling", icon="2️⃣")
    st.page_link("pages/3 Bidirected Network.py", label="Bidirected Network", icon="3️⃣")
    st.page_link("pages/4 Sunburst.py", label="Sunburst", icon="4️⃣")
    st.page_link("pages/5 Burst Detection.py", label="Burst Detection", icon="5️⃣")
    st.page_link("pages/6 Keywords Stem.py", label="Keywords Stem", icon="6️⃣")

st.header("Burst Detection", anchor=False)
st.subheader('Put your file here...', anchor=False)

#===clear cache===
def reset_all():
     st.cache_data.clear()

# Initialize NLP model
nlp = spacy.load("en_core_web_md")

@st.cache_data(ttl=3600)
def upload(extype):
    df = pd.read_csv(uploaded_file)
    #lens.org
    if 'Publication Year' in df.columns:
               df.rename(columns={'Publication Year': 'Year', 'Citing Works Count': 'Cited by',
                                     'Publication Type': 'Document Type', 'Source Title': 'Source title'}, inplace=True)
    return df

@st.cache_data(ttl=3600)
def get_ext(uploaded_file):
    extype = uploaded_file.name
    return extype

@st.cache_data(ttl=3600)
def get_minmax(df):
    MIN = int(df['Year'].min())
    MAX = int(df['Year'].max())
    GAP = MAX - MIN
    return MIN, MAX, GAP

@st.cache_data(ttl=3600)
def conv_txt(extype):
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DT': 'Document Type',
            'AB': 'Abstract',
            'PY': 'Year'}
    df = pd.read_csv(uploaded_file, sep='\t', lineterminator='\r')
    df.rename(columns=col_dict, inplace=True)
    return df

# Helper Functions
@st.cache_data(ttl=3600)
def get_column_name(df, possible_names):
    """Find and return existing column names from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the possible names {possible_names} found in DataFrame columns.")

@st.cache_data(ttl=3600)
def preprocess_text(text):
    """Lemmatize and remove stopwords from text."""
    return ' '.join([token.lemma_.lower() for token in nlp(text) if token.is_alpha and not token.is_stop])

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load data from the uploaded file."""
    extype = get_ext(uploaded_file)
    if extype.endswith('.csv'):
         df = upload(extype) 
    elif extype.endswith('.txt'):
         df = conv_txt(extype)

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
        
    if 'Title' in df.columns and 'Abstract' in df.columns:
        coldf = ['Abstract', 'Title']
    elif 'Title' in df.columns:
        coldf = ['Title']
    elif 'Abstract' in df.columns:
        coldf = ['Abstract']
    else:
        coldf = sorted(df.select_dtypes(include=['object']).columns.tolist())

    MIN, MAX, GAP = get_minmax(df)

    return df, coldf, MIN, MAX, GAP

@st.cache_data(ttl=3600)
def clean_data(df):

    years = list(range(YEAR[0],YEAR[1]+1))
    df = df.loc[df['Year'].isin(years)]
    
    # Preprocess text
    df['processed'] = df.apply(lambda row: preprocess_text(f"{row.get(col_name, '')}"), axis=1)
    
    # Vectorize processed text
    vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x.split())
    X = vectorizer.fit_transform(df['processed'].tolist())
    
    # Create DataFrame from the Document-Term Matrix (DTM)
    dtm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df['Year'].values)
    yearly_term_frequency = dtm.groupby(dtm.index).sum()

    # User inputs for top words analysis and exclusions
    excluded_words = [word.strip() for word in excluded_words_input.split(',')]
    
    # Identify top words, excluding specified words
    #top_words = [word for word in yearly_term_frequency.sum().nlargest(top_n).index if word not in excluded_words]
    filtered_words = [word for word in yearly_term_frequency.columns if word not in excluded_words]
    top_words = yearly_term_frequency[filtered_words].sum().nlargest(top_n).index.tolist()
    
    return yearly_term_frequency, top_words

@st.cache_data(ttl=3600)
def apply_burst_detection(top_words, data):
    all_bursts_list = []

    start_year = int(data.index.min())
    end_year = int(data.index.max())
    all_years = range(start_year, end_year + 1)
    
    continuous_years = pd.Series(index=all_years, data=0)  # Start with a series of zeros for all years

    years = continuous_years.index.tolist()
    
    all_freq_data = pd.DataFrame(index=years)
    
    for i, word in enumerate(top_words, start=1):
        # Update with actual counts where available
        word_counts = data[word].reindex(continuous_years.index, fill_value=0)
        
        # Convert years and counts to lists for burst detection
        r = continuous_years.index.tolist()  # List of all years
        r = np.array(r, dtype=int)
        d = word_counts.values.tolist()  # non-zero counts
        d = np.array(d, dtype=float)
        y = r.copy()
     
        if len(r) > 0 and len(d) > 0:
            n = len(r)
            q, d, r, p = burst_detection(d, r, n, s=2.0, gamma=1.0, smooth_win=1)
            bursts = enumerate_bursts(q, word)
            bursts = burst_weights(bursts, r, d, p)
            all_bursts_list.append(bursts)
    
            freq_data = yearly_term_frequency[word].reindex(years, fill_value=0)
            all_freq_data[word] = freq_data

    all_bursts = pd.concat(all_bursts_list, ignore_index=True)

    num_unique_labels = len(all_bursts['label'].unique())

    num_rows = math.ceil(top_n / num_columns)

    if running_total == "Running total":
        all_freq_data = all_freq_data.cumsum()
                        
    return all_bursts, all_freq_data, num_unique_labels, num_rows
      
# Streamlit UI for file upload
uploaded_file = st.file_uploader('', type=['csv', 'txt'], on_change=reset_all)

if uploaded_file is not None:
    try:
        c1, c2, c3 = st.columns([4,4,2])
        top_n = c1.number_input("Number of top words to analyze", min_value=1, value=9, step=1, on_change=reset_all)
        num_columns = c2.number_input("Number of columns for visualization", min_value=1, value=3, step=1, on_change=reset_all) 
        running_total = c3.selectbox("Option for counting words",
            ("Running total", "By occurrences each year"), on_change=reset_all)

        d1, d2 = st.columns([4,6])
        df, coldf, MIN, MAX, GAP = load_data(uploaded_file)
        col_name = d1.selectbox("Select column to analyze",
            (coldf), on_change=reset_all)
        excluded_words_input = d2.text_input("Words to exclude (comma-separated)", on_change=reset_all)

        if (GAP != 0):
            YEAR = st.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX), on_change=reset_all)
        else:
            st.write('You only have data in ', (MAX))
            sys.exit(1)
        
        yearly_term_frequency, top_words = clean_data(df) 
        
        bursts, freq_data, num_unique_labels, num_rows = apply_burst_detection(top_words, yearly_term_frequency)

        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])

        with tab1:        
            if bursts.empty:
                st.warning('We cannot detect any bursts', icon='⚠️')
    
            else:
                if num_unique_labels == top_n:
                    st.info(f'We detect a burst on {num_unique_labels} word(s)', icon="ℹ️")
                elif num_unique_labels < top_n:
                    st.info(f'We only detect a burst on {num_unique_labels} word(s), which is {top_n - num_unique_labels} fewer than the top word(s)', icon="ℹ️")
                
                fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=freq_data.columns[:top_n])
    
                row, col = 1, 1
                for i, column in enumerate(freq_data.columns[:top_n]):
                    fig.add_trace(go.Scatter(
                        x=freq_data.index, y=freq_data[column], mode='lines+markers+text', name=column,
                        line_shape='linear',
                        hoverinfo='text',
                        hovertext=[f"Year: {index}<br>Frequency: {freq}" for index, freq in zip(freq_data.index, freq_data[column])],
                        text=freq_data[column],
                        textposition='top center'
                    ), row=row, col=col)
                
                    # Add area charts
                    for _, row_data in bursts[bursts['label'] == column].iterrows():
                        x_values = freq_data.index[row_data['begin']:row_data['end']+1]
                        y_values = freq_data[column][row_data['begin']:row_data['end']+1]
                        
                        #middle_y = sum(y_values) / len(y_values)
                        y_post = min(freq_data[column]) + 1 if running_total == "Running total" else sum(y_values) / len(y_values)
                        x_offset = 0.1
                        
                        # Add area chart
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            fill='tozeroy', mode='lines', fillcolor='rgba(0,100,80,0.2)',
                        ), row=row, col=col)
    
                        align_value = "left" if running_total == "Running total" else "center"
                        valign_value = "bottom" if running_total == "Running total" else "middle"
                                            
                        # Add annotation for weight at the bottom
                        fig.add_annotation(
                            x=x_values[0] + x_offset,
                            y=y_post,
                            text=f"Weight: {row_data['weight']:.2f}",
                            showarrow=False,
                            font=dict(
                                color="black",
                                size=10
                            ),
                            align=align_value,
                            valign=valign_value,
                            textangle=270,
                            row=row, col=col
                        )
                
                    col += 1
                    if col > num_columns:
                        col = 1
                        row += 1
                
                fig.update_layout(
                    title_text="Scattertext",
                    showlegend=False,
                    height=num_rows * 400
                )
                
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
        with tab2:
            st.markdown('**Kleinberg, J. (2002). Bursty and hierarchical structure in streams. Knowledge Discovery and Data Mining.** https://doi.org/10.1145/775047.775061')

        with tab3:
            st.markdown('**Li, M., Zheng, Z., & Yi, Q. (2024). The landscape of hot topics and research frontiers in Kawasaki disease: scientometric analysis. Heliyon, 10(8), e29680–e29680.** https://doi.org/10.1016/j.heliyon.2024.e29680')
            st.markdown('**Domicián Máté, Ni Made Estiyanti and Novotny, A. (2024) ‘How to support innovative small firms? Bibliometric analysis and visualization of start-up incubation’, Journal of Innovation and Entrepreneurship, 13(1).** https://doi.org/10.1186/s13731-024-00361-z')
            st.markdown('**Lamba, M., Madhusudhan, M. (2022). Burst Detection. In: Text Mining for Information Professionals. Springer, Cham.** https://doi.org/10.1007/978-3-030-85085-2_6')
            
    except ValueError:
        st.error("An error occurred", icon="⚠️")
        sys.exit(1)
