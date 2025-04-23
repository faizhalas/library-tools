#import module
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
from textblob import TextBlob
import os
import numpy as np
import plotly.express as px
import json
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

@st.cache_resource(ttl=3600)
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
            
        c1, c2 = st.columns([6,4])
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
        
        wordcount = c2.number_input(label = "Words displayed", min_value = 0, step = 1)

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
            
            @st.cache_resource(ttl=3600)
            def remove_words(text):
                 words = text.split()
                 cleaned_words = [word for word in words if word not in remove_dict]
                 return ' '.join(cleaned_words) 
            
            paper['Sentences__'] = paper['Abstract_pre'].map(remove_words)

            return paper
        paper=clean_csv(extype) 
    
        if method == 'NLTKvader':
            analyzer = SentimentIntensityAnalyzer()

            @st.cache_resource(ttl=3600)
            def get_sentiment(text):
                score = analyzer.polarity_scores(text)
                return score

            tab1, tab2, tab3, tab4 = st.tabs(["📈 Result", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help"])
            with tab1:
                
                paper['Scores'] = paper['Sentences__'].apply(get_sentiment)

                scoreframe = pd.DataFrame()

                scoreframe['Phrase'] = pd.Series(paper['Sentences__'])

                scoreframe[['Negativity','Neutrality','Positivity','Compound']] = pd.DataFrame.from_records(paper['Scores'])

                scoreframe = scoreframe.groupby(scoreframe.columns.tolist(),as_index=False).size()

                scoreframe = scoreframe.truncate(after = wordcount)

                with st.expander("Sentence and Results"):
                    finalframe = pd.DataFrame()
                    finalframe['Sentence'] = scoreframe['Phrase']
                    finalframe[['Negativity','Neutrality','Positivity','Compound']] = scoreframe[['Negativity','Neutrality','Positivity','Compound']]
                    finalframe[['Count']] = scoreframe[['size']]

                    st.dataframe(finalframe, use_container_width=True)

            with tab2:
                st.markdown('**Hutto, C. and Gilbert, E. (2014) ‘VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text’, Proceedings of the International AAAI Conference on Web and Social Media, 8(1), pp. 216–225.** https://doi.org/10.1609/icwsm.v8i1.14550')

            with tab3:
                st.markdown('**Author** URL')
                st.markdown('**Author** URL')

            with tab4:
                st.write('Empty')
        
        elif(method == 'TextBlob'):
            
            @st.cache_resource(ttl=3600)
            def get_sentimentb(text):
                line = TextBlob(text)
                return line.sentiment

            @st.cache_resource(ttl=3600)
            def get_assessments(frame):
                text = TextBlob(str(frame))

                polar, subject, assessment = text.sentiment_assessments

                try:
                    phrase, phrasepolar, phrasesubject, unknown = assessment[0]
                except: #this only happens if assessment is empty
                    phrase, phrasepolar, phrasesubject = "empty", 0, 0

                return phrase, phrasepolar, phrasesubject

            @st.cache_resource(ttl=3600)
            def mergelist(data):
                return ' '.join(data)

            @st.cache_resource(ttl=3600)
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

            neut = phraseframe.loc[phraseframe['Score']=="Neutral"]
            neut.reset_index(inplace = True)

            pos = phraseframe.loc[phraseframe['Score']=="Positive"]
            pos.reset_index(inplace = True)

            neg = phraseframe.loc[phraseframe['Score']=="Negative"]
            neg.reset_index(inplace = True)

            paper['Sentiment'] = paper['Sentences__'].apply(get_sentimentb)

            pos.sort_values(by=["size"], inplace = True, ascending = False, ignore_index = True)
            pos = pos.truncate(after = wordcount)

            neg.sort_values(by=["size"], inplace = True, ascending = False, ignore_index = True)
            neg = neg.truncate(after = wordcount)
        
            neut.sort_values(by=["size"], inplace = True, ascending = False, ignore_index = True)
            neut = neut.truncate(after = wordcount)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help", "Interpreting Results"])
            with tab1:
                #display tables and graphs
    
                with st.expander("Positive Sentiment"):
                    st.dataframe(pos, use_container_width=True)
                    figpos = px.bar(pos, x="Phrase", y="size", labels={"size": "Count", "Phrase": "Word"})      
                    st.plotly_chart(figpos, use_container_width=True)
    
                with st.expander("Negative Sentiment"):
                    st.dataframe(neg, use_container_width=True)
                    figneg = px.bar(neg, x="Phrase", y="size", labels={"size": "Count", "Phrase": "Word"}, color_discrete_sequence=["#e57d7d"])
                    st.plotly_chart(figneg, use_container_width=True)
    
                with st.expander("Neutral Sentiment"):
                    st.dataframe(neut, use_container_width=True)
                    figneut = px.bar(neut, x="Phrase", y="size", labels={"size": "Count", "Phrase": "Word"}, color_discrete_sequence=["#737a72"])
                    st.plotly_chart(figneut, use_container_width=True)


                with st.expander("Sentence and Results"):
                    finalframe = pd.DataFrame()
                    finalframe['Sentence'] = paper['Sentences__']
                    finalframe[['Polarity','Subjectivity']] = pd.DataFrame(paper['Sentiment'].tolist(), index = paper.index)
            
                    st.dataframe(finalframe, use_container_width=True)

            with tab2:
                st.markdown('**Steven, L. et al. (2018) TextBlob: Simplified Text Processing — TextBlob 0.15.2 documentation, Readthedocs.io.** https://textblob.readthedocs.io/en/dev/')

            with tab3:
                st.markdown('**Author** URL')
                st.markdown('**Author** URL')

            with tab4:
                st.write('Empty')

            with tab5:
                st.header("TextBlob")
                st.write("Polarity represents positive, negative, and neutral tone, and is between [-1, 1]. -1 is very negative, 1 is very positive")
                st.write("Subjectivity represents objectiveness and subjectiveness, and is between [0, 1]. 0 is very objective, 1 is very subjective.")
                
                st.header("NLTKVader")


    except Exception as e:
        st.write(e)
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
