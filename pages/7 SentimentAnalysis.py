#import module
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import nltk
import pandas as pd
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
from textblob import TextBlob
import os
import matplotlib.pyplot as plt
import numpy as np

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


st.header("Sentiment Analysis", anchor=False)
st.subheader('Put your file here...', anchor=False)

#========unique id========
@st.cache_resource(ttl=3600)
def create_list():
    l = [1, 2, 3]
    return l

l = create_list()
first_list_value = l[0]
l[0] = first_list_value + 1
uID = str(l[0])

@st.cache_data(ttl=3600)
def get_ext(uploaded_file):
    extype = uID+uploaded_file.name
    return extype

#===clear cache===


def reset_all():
    st.cache_data.clear()

#===avoiding deadlock===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
#===upload file===
@st.cache_data(ttl=3600)
def upload(file):
    papers = pd.read_csv(uploaded_file)
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
    keywords = pd.read_json(uploaded_file)
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

        coldf = sorted(papers.select_dtypes(include=['object']).columns.tolist())
            
        c1, c2 = st.columns([3,4])
        ColCho = c1.selectbox(
                'Choose column',
                (coldf), on_change=reset_all)
        method = c2.selectbox(
            'Choose method',[
            'TextBlob','NLTKvader']
        )
        words_to_remove = c1.text_input("Remove specific words. Separate words by semicolons (;)")
        rem_copyright = c2.toggle('Remove copyright statement', value=True, on_change=reset_all)
        rem_punc = c2.toggle('Remove punctuation', value=True, on_change=reset_all)
         
        #===clean csv===
        @st.cache_data(ttl=3600, show_spinner=False)
        def clean_csv(extype):
            paper = papers.dropna(subset=[ColCho])
                     
            #===mapping===
            paper['Abstract_pre'] = paper[ColCho].map(lambda x: x.lower())
            if rem_punc:
                 paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('[,:;\.!-?•=]', ' ', x))
                 paper['Abstract_pre'] = paper['Abstract_pre'].str.replace('\u201c|\u201d', '', regex=True) 
            if rem_copyright:  
                 paper['Abstract_pre'] = paper['Abstract_pre'].map(lambda x: re.sub('©.*', '', x))
            
            #===stopword removal===
            stop = stopwords.words('english')
            paper[ColCho] = paper['Abstract_pre'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
              
            words_rmv = [word.strip() for word in words_to_remove.split(";")]
            remove_dict = {word: None for word in words_rmv}
            def remove_words(text):
                 words = text.split()
                 cleaned_words = [word for word in words if word not in remove_dict]
                 return ' '.join(cleaned_words) 
            paper['Sentences__'] = paper['Abstract_pre'].map(remove_words)

            return paper
        paper=clean_csv(extype) 
    
        if method == 'NLTKvader':
            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment(text):
                score = analyzer.polarity_scores(text)
                return score
            paper['Sentiment'] = paper['Sentences__'].apply(get_sentiment)
        
        elif(method == 'TextBlob'):
            
            def get_sentimentb(text):
                line = TextBlob(text)
                return line.sentiment

            def get_assessments(frame):
                text = TextBlob(str(frame))

                polar, subject, assessment = text.sentiment_assessments

                try:
                    phrase, phrasepolar, phrasesubject, unknown = assessment[0]
                except: #this only happens if assessment is empty
                    phrase, phrasepolar, phrasesubject = "empty", 0, 0

                return phrase, phrasepolar, phrasesubject

            def mergelist(data):
                return ' '.join(data)

            def assignscore(data):
                if data>0:
                    return "Positive"
                elif data<0:
                    return "Negative"
                else:
                    return "Neutral"

            phrases = paper['Sentences__'].apply(get_assessments)

            phraselist = phrases.to_list()

            phraseframe = pd.DataFrame(phraselist, columns =["Phrase","Polarity","Subjectivity"])

            phraseframe["Phrase"] = phraseframe["Phrase"].apply(mergelist)

            phraseframe = phraseframe.groupby(phraseframe.columns.tolist(),as_index=False).size()

            phraseframe["Score"] = phraseframe["Polarity"].apply(assignscore)

            neut = phraseframe.drop(phraseframe[phraseframe['Score']=='Positive'].index)

            neut = neut.drop(neut[neut['Score']=='Negative'].index)

            frame = phraseframe.drop(phraseframe[phraseframe['Score']=='Neutral'].index)

            pos = frame.drop(frame[frame['Score']=='Negative'].index)

            neg = frame.drop(frame[frame['Score']=='Positive'].index)



            paper['Sentiment'] = paper['Sentences__'].apply(get_sentimentb)

            pos.sort_values(by=["size"], inplace = True, ascending = False, ignore_index = True)

            pos = pos.truncate(after = 10)


            neg.sort_values(by=["size"], inplace = True, ascending = False, ignore_index = True)
            
            neg = neg.truncate(after = 10)
        
            neut.sort_values(by=["size"], inplace = True, ascending = False, ignore_index = True)

            neut = neut.truncate(after = 10)

            #display tables and graphs

            st.header("Positive Sentiment")

            st.dataframe(pos)

            st.bar_chart(pos, x = "Phrase", y = "size", y_label = "Word", x_label = "Count", horizontal = True)

            st.header("Negative Sentiment")

            st.dataframe(neg)

            st.bar_chart(neg, x = "Phrase", y = "size", y_label = "Word", x_label = "Count", horizontal = True, color = "#e57d7d")

            st.header("Neutral Sentiment")

            st.dataframe(neut)

            st.bar_chart(neut, x = "Phrase", y = "size", y_label = "Word", x_label = "Count", horizontal = True)

        st.header("Sentence and Results")

        finalframe = pd.DataFrame()
        finalframe['Sentence'] = paper['Sentences__']
        finalframe['Sentiment'] = paper['Sentiment']


        st.dataframe(finalframe)



    except:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
