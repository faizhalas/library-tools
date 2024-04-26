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

# Initialize NLP model
nlp = spacy.load("en_core_web_md")

# Directory for saving plots
plot_directory = "plots"
os.makedirs(plot_directory, exist_ok=True)

# Helper Functions
def get_column_name(df, possible_names):
    """Find and return existing column names from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the possible names {possible_names} found in DataFrame columns.")

def preprocess_text(text):
    """Lemmatize and remove stopwords from text."""
    return ' '.join([token.lemma_.lower() for token in nlp(text) if token.is_alpha and not token.is_stop])

def apply_burst_detection(word, data):
    """Apply burst detection for a given word"""
    start_year = int(data.index.min())
    end_year = int(data.index.max())
    all_years = range(start_year, end_year + 1)

    continuous_years = pd.Series(index=all_years, data=0)  # Start with a series of zeros for all years
    
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
        q, d, r, p= burst_detection(d, r, n, s=2.0, gamma=1.0, smooth_win=1)
        bursts = enumerate_bursts(q, word)
        weighted_bursts = burst_weights(bursts,r,d,p)
        return bursts, y
    else:
        return pd.DataFrame(), y 


@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load data from the uploaded file."""
    df = pd.read_csv(uploaded_file)
    # Dynamic column naming based on common identifiers in different datasets
    column_mapping = {
        'TI': 'Title', 'AB': 'Abstract', 'Year': 'Year',
        'Publication Year': 'Year'
    }
    df.rename(columns={original: standard for original, standard in column_mapping.items() if original in df.columns}, inplace=True)
    return df

# Streamlit UI for file upload
uploaded_file = st.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = load_data(uploaded_file)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df.sort_values(by='Year', inplace=True)
        
        # Preprocess text
        df['processed'] = df.apply(lambda row: preprocess_text(f"{row.get('Title', '')} {row.get('Abstract', '')}"), axis=1)
    
        # Vectorize processed text
        vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x.split())
        X = vectorizer.fit_transform(df['processed'].tolist())
    
        # Create DataFrame from the Document-Term Matrix (DTM)
        dtm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df['Year'].values)
        yearly_term_frequency = dtm.groupby(dtm.index).sum()
    
        # User inputs for top words analysis and exclusions
        top_n = st.number_input("Number of top words to analyze", min_value=1, value=6, step=1)
        excluded_words_input = st.text_input("Words to exclude (comma-separated)")
        excluded_words = [word.strip() for word in excluded_words_input.split(',')]
    
        # Identify top words, excluding specified words
        top_words = [word for word in yearly_term_frequency.sum().nlargest(top_n).index if word not in excluded_words]
    
        # Generate and display burst detection plots
        cols = 3
        rows = math.ceil(len(top_words) / cols)
        # plt.figure(figsize=(20, 6 * rows))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
        count = 0
        for i, word in enumerate(top_words, start=1):
            fig, ax = plt.subplots(figsize=(20, 6))#rows, cols, i)
            bursts, years = apply_burst_detection(word, yearly_term_frequency)
            
            if not bursts.empty:
                freq_data = yearly_term_frequency[word].reindex(years, fill_value=0)
                ax.plot(years.astype(int), freq_data, marker='o', label=f'"{word}" frequency')
                ax.set_xticks(range(int(min(years)), int(max(years)) + 1))
                ax.set_title(f'Burst periods for "{word}"')
                ax.set_xlabel('Year')
                ax.set_ylabel('Frequency')
                label_offset = max(freq_data) * 0.02
                #adding value label
                for year, value in zip(years, freq_data):
                    ax.text(year, value + label_offset, f'{value:.2f}', ha='center', va='bottom')
                for _, row in bursts.iterrows():
                    begin_year = int(years[int(row['begin'])])
                    end_year = int(years[int(row['end'])])   
                    burst_range = np.arange(begin_year, end_year + 1)
                    burst_freq = freq_data.loc[burst_range]
                    ax.fill_between(burst_range, 0, burst_freq, color='orange', alpha=0.5)
            #Adjust layout to prevent overlap and save the entire figure
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                count+=1
        if count > 0:
            st.warning(f'No bursts detected')
    else:
        st.error("Unsupported file type.")
else:
    st.info("Please upload a file to begin.")
