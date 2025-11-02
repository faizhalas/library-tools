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
import io
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
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
    st.page_link("pages/7 Sentiment Analysis.py", label="Sentiment Analysis", icon="7️⃣")
    st.page_link("pages/8 Shifterator.py", label="Shifterator", icon="8️⃣")
    st.page_link("pages/9 WordCloud.py", label = "WordCloud", icon = "9️⃣")

with st.expander("Before you start", expanded = True):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download Visualization"])
    with tab1:
        st.write("Burst detection identifies periods when a specific event occurs with unusually high frequency, referred to as 'bursty'. This method can be applied to identify bursts in a continuous stream of events or in discrete groups of events (such as poster title submissions to an annual conference).") 
        st.divider()
        st.write('💡 The idea came from this:') 
        st.write('Kleinberg, J. (2002). Bursty and hierarchical structure in streams. Knowledge Discovery and Data Mining. https://doi.org/10.1145/775047.775061')
                    
    with tab2:
        st.text("1. Put your file. Choose your preferred column to analyze.")
        st.text("2. Choose your preferred method to compare.")
        st.text("3. Finally, you can visualize your data.")
        st.error("This app includes lemmatization and stopwords. Currently, we only offer English words.", icon="💬")

    with tab3:
        st.code("""
        +----------------+------------------------+----------------------------------+
        |     Source     |       File Type        |              Column              |
        +----------------+------------------------+----------------------------------+
        | Scopus         | Comma-separated values | Choose your preferred column     |
        |                | (.csv)                 | that you have to analyze and     |
        +----------------+------------------------| and need a column called "Year"  |
        | Web of Science | Tab delimited file     |                                  |
        |                | (.txt)                 |                                  |
        +----------------+------------------------|                                  |
        | Lens.org       | Comma-separated values |                                  |
        |                | (.csv)                 |                                  |
        +----------------+------------------------|                                  |
        | Dimensions     | Comma-separated values |                                  |
        |                | (.csv)                 |                                  |
        +----------------+------------------------|                                  |
        | OpenAlex       | Comma-separated values |                                  |
        |                | (.csv)                 |                                  |
        +----------------+------------------------|                                  |
        | Other          | .csv .xls .xlsx        |                                  |
        +----------------+------------------------|                                  |
        | Hathitrust     | .json                  |                                  |
        +----------------+------------------------+----------------------------------+
        """, language=None)
                
    with tab4:
        st.subheader(':blue[Burst Detection]', anchor=False)
        st.button('📊 Download high resolution image.')
        st.text("Click download button.") 

        st.divider()
        st.subheader(':blue[Top words]', anchor=False)
        st.button('👉 Click to download list of top words.')
        st.text("Click download button.")  

        st.divider()
        st.subheader(':blue[Burst]', anchor=False)
        st.button('👉 Click to download the list of detected bursts.')
        st.text("Click download button.") 

st.header("Burst Detection", anchor=False)
st.subheader('Put your file here...', anchor=False)

#===clear cache===
def reset_all():
    st.cache_data.clear()

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

@st.cache_data(ttl=3600)
def upload(extype):
    df = pd.read_csv(uploaded_file)
    #lens.org
    if 'Publication Year' in df.columns:
               df.rename(columns={'Publication Year': 'Year', 'Citing Works Count': 'Cited by',
                                  'Publication Type': 'Document Type', 'Source Title': 'Source title'}, inplace=True)
        
    elif "About the data" in df.columns[0]:
        df = sf.dim(df)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        df.rename(columns=col_dict, inplace=True)
        
    elif "ids.openalex" in df.columns:
        df.rename(columns={'publication_year': 'Year'}, inplace=True)
        
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
    elif extype.endswith('.json'):
        df = conv_json(extype)
    elif extype.endswith('.tar.gz') or extype.endswith('.xml'):
        df = conv_pub(uploaded_file)
    elif extype.endswith(('.xls', '.xlsx')):
        df = readxls(extype)

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

    ngram_range = (1, xgram)
    
    # Vectorize processed text
    if count_method == "Document Frequency":
        vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x.split(), binary=True, ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x.split(), ngram_range=ngram_range)
    X = vectorizer.fit_transform(df['processed'].tolist())
    
    # Create DataFrame from the Document-Term Matrix (DTM)
    dtm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df['Year'].values)
    yearly_term_frequency = dtm.groupby(dtm.index).sum()

    # excluded & included words
    if exc_inc == "Words to exclude":
        excluded_words = [word.strip() for word in words_input.split(',')]
        filtered_words = [word for word in yearly_term_frequency.columns if word not in excluded_words]
    
    elif exc_inc == "Focus on these words":
        included_words = [word.strip() for word in words_input.split(',')]   
        filtered_words = [word for word in yearly_term_frequency.columns if word in included_words]

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

    num_rows = math.ceil(top_n / 2)

    if running_total == "Running total":
        all_freq_data = all_freq_data.cumsum()
                        
    return all_bursts, all_freq_data, num_unique_labels, num_rows

@st.cache_data(ttl=3600)
def convert_df(df):
    return df.to_csv().encode("utf-8")

@st.cache_data(ttl=3600)
def scattervis(bursts, freq_data, top_n):
    freq_data = freq_data.reset_index()
    freq_data.rename(columns={"index": "Year"}, inplace=True)
    
    freq_data_melted = freq_data.melt(id_vars=["Year"], var_name="Category", value_name="Value")
    freq_data_melted = freq_data_melted[freq_data_melted["Value"] > 0]
    
    wordlist = freq_data_melted["Category"].unique()
    years = freq_data["Year"].tolist()
    
    bursts["begin"] = bursts["begin"].apply(lambda x: years[min(x, len(years) - 1)] if x < len(years) else None)
    bursts["end"] = bursts["end"].apply(lambda x: years[min(x, len(years) - 1)] if x < len(years) else None)

    burst_points = []
    for _, row in bursts.iterrows():
        for year in range(row["begin"], row["end"] + 1):
            burst_points.append((year, row["label"], row["weight"]))
    burst_points_df = pd.DataFrame(burst_points, columns=["Year", "Category", "Weight"])

    min_year = min(years)
    max_year = max(years)
    n_years = max_year - min_year + 1
    n_labels = len(wordlist)

    label_spacing = 50   
    year_spacing = 60    

    plot_height = n_labels * label_spacing + 100
    plot_width = n_years * year_spacing + 150

    fig = go.Figure()

    # scatter trace for burst points
    fig.add_trace(go.Scatter(
        x=burst_points_df["Year"],
        y=burst_points_df["Category"],
        mode='markers',
        marker=dict(
            symbol='square',
            size=40,
            color='red',
            opacity=0.5
        ),
        hoverinfo='text',
        text=burst_points_df["Weight"],
        showlegend=False
    ))

    # scatter trace for freq_data
    fig.add_trace(go.Scatter(
        x=freq_data_melted["Year"],
        y=freq_data_melted["Category"],
        mode='markers+text',
        marker=dict(
            symbol='square',
            size=30,
            color=freq_data_melted["Value"],
            colorscale='Blues',
            showscale=False
        ),
        text=freq_data_melted["Value"],
        textposition="middle center",
        textfont=dict(
            size=16,
            color=['white' if value > freq_data_melted["Value"].max()/2 else 'black'
                   for value in freq_data_melted["Value"]]
        )
    ))

    # Layout
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[min_year - 1, max_year + 1],
            tickfont=dict(size=16),
            automargin=True,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            tickvals=wordlist,
            ticktext=wordlist,
            tickmode='array',
            tickfont=dict(size=16),
            automargin=True,
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=plot_height,
        width=plot_width,
        autosize=False
    )
                    
    fig.write_image("scatter_plot.png", width=plot_width, height=plot_height)
    st.image("scatter_plot.png", use_column_width=False)
        
    pio.write_image(fig, 'result.png', width=plot_width, height=plot_height, scale=4)  

@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def linegraph(bursts, freq_data, top_n, running_total=""):
    num_rows = (top_n + 1) // 2   # 2 columns layout

    # --- X spacing: each year gets a slot of width 10 (=> ±5 padding) ---
    years = list(freq_data.index)
    spacing = 100
    padding = 200
    x_positions = np.arange(len(years)) * spacing      # 0,10,20,...
    tickvals = x_positions
    ticktext = [str(y) for y in years]

    fig = make_subplots(
        rows=num_rows,
        cols=2,
        subplot_titles=freq_data.columns[:top_n]
    )

    row, col = 1, 1
    for i, column in enumerate(freq_data.columns[:top_n]):
        # main line (x mapped to spaced positions)
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=freq_data[column].to_numpy(),
            mode='lines+markers+text',
            name=column,
            line_shape='linear',
            hoverinfo='text',
            hovertext=[f"Year: {yr}<br>Frequency: {freq}"
                       for yr, freq in zip(years, freq_data[column])],
            text=freq_data[column],
            textposition='top center'
        ), row=row, col=col)

        # bursts shading + annotation
        for _, row_data in bursts[bursts['label'] == column].iterrows():
            # slice by positional indices (begin/end are positions)
            x_vals = x_positions[row_data['begin']:row_data['end'] + 1]
            y_vals = freq_data[column].iloc[row_data['begin']:row_data['end'] + 1].to_numpy()

            # area under the line during burst
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                fill='tozeroy',
                mode='lines',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(width=0)  # keep it as a filled area
            ), row=row, col=col)

            # weight label near the bottom
            y_post = float(np.nanmin(freq_data[column])) * 0.95
            x_offset = 0.5  # small shift within the 10-wide slot

            align_value = "left" if running_total == "Running total" else "center"
            valign_value = "bottom" if running_total == "Running total" else "middle"

            fig.add_annotation(
                x=x_vals[0] + x_offset,
                y=y_post,
                text=f"Weight: {row_data['weight']:.2f}",
                showarrow=False,
                font=dict(color="black", size=12),
                align=align_value,
                valign=valign_value,
                textangle=270,
                row=row, col=col)

        col += 1
        if col > 2:
            col = 1
            row += 1

    # Dynamic sizing
    plot_height = num_rows * 500
    plot_width = len(years) * spacing + padding

    # Apply the same x settings to all subplots:
    fig.update_xaxes(
        range=[-spacing/2, x_positions[-1] + spacing/2],  # ±5 around the ends
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=100, b=20),
        height=plot_height,
        width=plot_width,
        autosize=False
    )
                
    fig.write_image("line_graph.png", width=plot_width, height=plot_height)
    
    st.image("line_graph.png", use_column_width=False)
    pio.write_image(fig, 'result.png', width=plot_width, height=plot_height, scale=4)

@st.cache_data(ttl=3600)
def download_result(freq_data, bursts):
    csv1 = convert_df(freq_data)
    csv2 = convert_df(bursts)
    return csv1, csv2
      
uploaded_file = st.file_uploader('', type=['csv', 'txt', 'json', 'tar.gz', 'xml', 'xls', 'xlsx'], on_change=reset_all)

if uploaded_file is not None:
    try:
        c1, c2, c3 = st.columns([3,3,4])
        top_n = c1.number_input("Number of top words to analyze", min_value=5, value=10, step=1, on_change=reset_all)
        viz_selected = c2.selectbox("Option for visualization",
            ("Line graph", "Heatmap"), on_change=reset_all)
        running_total = c3.selectbox("Calculation method",
            ("Running total", "By occurrences each year"), on_change=reset_all)
        count_method = c1.selectbox("Count by",
            ("Term Frequency", "Document Frequency"), on_change=reset_all)

        df, coldf, MIN, MAX, GAP = load_data(uploaded_file)
        col_name = c2.selectbox("Select column to analyze",
            (coldf), on_change=reset_all)
        xgram = c3.selectbox("N-grams", ("1", "2", "3"), on_change=reset_all)
        xgram = int(xgram)

        st.divider()
        d1, d2 = st.columns([3,7])
        exc_inc = d1.radio("Select to exclude or focus on specific words", ["Words to exclude","Focus on these words"], horizontal=True, on_change=reset_all)
        words_input = d2.text_input("Words to exclude or focus on (comma-separated)", on_change=reset_all)

        if (GAP != 0):
            YEAR = st.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX), on_change=reset_all)
        else:
            c1.write('You only have data in ', (MAX))
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

                if viz_selected == "Line graph": 
                    linegraph(bursts, freq_data, top_n)
                    
                elif viz_selected =="Heatmap":
                    scattervis(bursts, freq_data, top_n)
                
                csv1, csv2 = download_result(freq_data, bursts)
                e1, e2, e3 = st.columns(3)
                with open('result.png', "rb") as file:
                    btn = e1.download_button(
                        label="📊 Download high resolution image",
                        data=file,
                        file_name="burst.png",
                        mime="image/png")
                    
                e2.download_button(
                    "👉 Click to download list of top words",
                    csv1,
                    "top-keywords.csv",
                    "text/csv")
    
                e3.download_button(
                    "👉 Click to download the list of detected bursts",
                    csv2,
                    "burst.csv",
                    "text/csv")
 
        with tab2:
            st.markdown('**Kleinberg, J. (2002). Bursty and hierarchical structure in streams. Knowledge Discovery and Data Mining.** https://doi.org/10.1145/775047.775061')

        with tab3:
            st.markdown('**Li, M., Zheng, Z., & Yi, Q. (2024). The landscape of hot topics and research frontiers in Kawasaki disease: scientometric analysis. Heliyon, 10(8), e29680–e29680.** https://doi.org/10.1016/j.heliyon.2024.e29680')
            st.markdown('**Domicián Máté, Ni Made Estiyanti and Novotny, A. (2024) ‘How to support innovative small firms? Bibliometric analysis and visualization of start-up incubation’, Journal of Innovation and Entrepreneurship, 13(1).** https://doi.org/10.1186/s13731-024-00361-z')
            st.markdown('**Lamba, M., Madhusudhan, M. (2022). Burst Detection. In: Text Mining for Information Professionals. Springer, Cham.** https://doi.org/10.1007/978-3-030-85085-2_6')
            st.markdown('**Santosa, F. A. (2025). Artificial Intelligence in Library Studies: A Textual Analysis. JLIS.It, 16(1).** https://doi.org/10.36253/jlis.it-626')
                     
    except Exception as e:
        st.error("Please ensure that your file or settings are correct. If you think there is a mistake, feel free to reach out to us!", icon="🚨")
        st.stop()
