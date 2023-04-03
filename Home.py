#import module
import streamlit as st
from PIL import Image

#===config===
st.set_page_config(
     page_title="Coconut",
     page_icon="🥥",
     layout="wide"
)
st.title('🥥 Coconut Library Tools')
st.sidebar.success('Select page above')

#===page===
tab1, tab2, tab3 = st.tabs(["About", "How to", "Behind this app"])

with tab1:
   st.header("🌌 Hello universe!")
   st.write('The coconut tree is known as one of the most useful trees. 🌴 The leaves function to produce oxygen through photosynthesis and are used for handicrafts, even for roof houses. The shells, the oil, the wood, the flowers, or even the husks can be something useful. From this philosophy, the Coconut Library Tool aims to be useful for librarians or anyone who needs cool features but is hindered by programming languages.')
   st.write("We thank the cool people who have created so many facilities that we can place them in a place. We can't name them all, but we believe science will advance due to your efforts. 🧑🏻‍🤝‍🧑🏾")
   st.text('')
   st.text('')
   st.text('')
   st.error("Currently, this app only works on Scopus's CSV file.", icon="🚨")
         
with tab2:
   st.header("Before you start")
   option = st.selectbox(
    'Please choose....',
    ('Keyword Stem', 'Topic Modeling', 'Association Rules', 'Sunburst'))
   
   if option == 'Keyword Stem':
        st.write('💡 The idea came from this:')
        st.write('(Published soon) Santosa, F. A. (2022). Prior steps into knowledge mapping: Text mining application and comparison. Issues in Science and Technology Librarianship, 102. https://doi.org/10.29173/istl2736')
        if st.button('🌟 Show me'):
            st.text("1. Put your Scopus CSV file.")
            st.text("2. Choose your preferable method. Picture below may help you to choose wisely.")
            st.markdown("![Source: https://studymachinelearning.com/stemming-and-lemmatization/](https://studymachinelearning.com/wp-content/uploads/2019/09/stemmin_lemm_ex-1.png)")
            st.text('Source: https://studymachinelearning.com/stemming-and-lemmatization/')
            st.text("3. Now you need to select what kind of keywords you need.")
            st.text("4. Finally, you can download and use the file on VOSviewer, Bibliometrix, or else!")
            st.error("Please check what has changed. It's possible some keywords failed to find their roots.", icon="🚨")
            
   elif option == 'Topic Modeling':
        st.write('💡 The idea came from this:')
        st.write('Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing and interpreting topics. In Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces. Association for Computational Linguistics. https://doi.org/10.3115/v1/w14-3110')
        if st.button('🌟 Show me'):
            st.text("1. Put your Scopus CSV file. We use abstract column for this process.")
            st.text("2. Click calculate coherence to know the best score for your data.")
            st.text("3. Finally, you can visualize your data.")
            st.error("This app includes lemmatization and stopwords for the abstract text. Currently, we only offer English words. For other languages you can use stemming.", icon="💬")
            st.error("If you want to see the topic on another data (chats, questionnaire, or other text), change the column name of your table to 'Abstract' or use the other tool that we offer.", icon="🚨")
                         
   elif option == 'Bidirected Network':
        st.write('💡 The idea came from this:') 
        st.write('Munan Li (2016): An exploration to visualise the emerging trends of technology foresight based on an improved technique of co-word analysis and relevant literature data of WOS, Technology Analysis & Strategic Management, DOI: 10.1080/09537325.2016.1220518')
        st.write('Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. In ACM SIGMOD Record (Vol. 22, Issue 2, pp. 207–216). Association for Computing Machinery (ACM). https://doi.org/10.1145/170036.170072')
        if st.button('🌟 Show me'):
            st.text("1. Put your Scopus CSV file.")
            st.text("2. Choose your preferable method. Picture below may help you to choose wisely.")
            st.markdown("![Source: https://studymachinelearning.com/stemming-and-lemmatization/](https://studymachinelearning.com/wp-content/uploads/2019/09/stemmin_lemm_ex-1.png)")
            st.text('Source: https://studymachinelearning.com/stemming-and-lemmatization/')
            st.text("3. Choose the value of Support and Confidence. If you're not sure how to use it please read the article above or just try it!")
            st.text("4. You can see the table and a simple visualization before making a network visualization.")
            st.text('5. Click "Generate network visualization" to see the network')
            st.error("The more data on your table, the more you'll see on network.", icon="🚨")
            st.error("If the table contains many rows, the network will take more time to process. Please use it efficiently.", icon="⌛")
    
   elif option == 'Sunburst':
        st.text("1. Put your Scopus CSV file.")
        st.text("2. You can set the range of years to see how it changed.")
        st.text("3. The sunburst has 3 levels. The inner circle is the type of data, meanwhile, the middle is the source title and the outer is the year the article was published.")
        st.text("4. The size of the slice depends on total documents. The average of inner and middle levels is calculated by formula below:")
        st.code('avg = sum(a * weights) / sum(weights)', language='python')
     
with tab3:
   st.header('Behind this app')
   st.subheader('Faizhal Arif Santosa')
   st.text('Librarian. National Research and Innovation Agency.')
   st.text('')
   st.subheader('Crissandra George')
   st.text('Aspiring Academic Librarian and Archivist. University of Kentucky.')
   st.text('')
   st.text('')
   st.text('If you want to take a part, please let us know!')
     
     
