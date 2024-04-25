import scattertext as sct
import spacy
import pandas as pd
import en_core_web_md
import streamlit as st


# load language model
nlp = en_core_web_md.load()
nlp = spacy.load("en_core_web_md")

# Scopus file loading
st.title("Scattertext Analysis")
st.header("Put your file here... ")

def compatison1(selected_column):
    # type_of_comparison 1
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        first_source = st.selectbox("Choose First Source", df[source_title_col].unique(), key='first_source_select')
    with row2_col2:
        second_source = st.selectbox("Choose Second Source", df[source_title_col].unique(),
                                     key='second_source_select')

    # filter data
    first_data = df[df[source_title_col] == first_source].copy()
    second_data = df[df[source_title_col] == second_source].copy()
    filtered_data = pd.concat([first_data, second_data])

    if st.button("Generate the Scattertext Plot"):
        # make plot
        corpus = sct.CorpusFromPandas(
            filtered_data,
            category_col= source_title_col,
            text_col= selected_column,
            nlp=nlp,
        ).build()
        # generate HTML visualization
        html = sct.produce_scattertext_explorer(corpus,
                                                category=first_source,
                                                category_name=first_source,
                                                not_category_name=second_source,
                                                width_in_pixels=900,
                                                minimum_term_frequency=0,
                                                metadata=filtered_data)
        st.components.v1.html(html, width=1000, height=600)
    return


        # type_of_comparison 2
def comparison2(selected_column):
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df.dropna(subset=[year_col], inplace=True)
    df[year_col] = df[year_col].astype(int)

    min_year = int(df[year_col].min())
    max_year = int(df[year_col].max())
    # layout row2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        first_range = st.slider("First range", min_value = min_year, max_value= max_year, step = 1, value= (min_year, max_year))
    with row2_col2:
        second_range = st.slider("Second range", min_value = min_year, max_value= max_year, step = 1, value= (min_year, max_year))

    # filter data
    first_range_filter_df = df[(df[year_col] >= first_range[0]) & (df[year_col] <= first_range[1])].copy()
    first_range_filter_df['Topic Range'] = 'First range'

    second_range_filter_df = df[(df[year_col] >= second_range[0]) & (df[year_col] <= second_range[1])].copy()
    second_range_filter_df['Topic Range'] = 'Second range'

    filtered_df = pd.concat([first_range_filter_df, second_range_filter_df])

    if st.button("Generate the Scattertext Plot"):
        # make plot
        corpus = sct.CorpusFromPandas(
            filtered_df,
            category_col="Topic Range",
            text_col= selected_column,
            nlp=nlp,
        ).build()
        # generate HTML visualization
        html = sct.produce_scattertext_explorer(corpus,
                                                category='First range',
                                                category_name='First range',
                                                not_category_name='Second range',
                                                width_in_pixels=900,
                                                minimum_term_frequency=0,
                                                metadata=filtered_df)
        st.components.v1.html(html, width=1000, height=600)
    return


if __name__ == '__main__':
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])
    if uploaded_file is not None:
        # determine file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            abstract_col = 'Abstract'
            title_col = 'Title'
            source_title_col = 'Source title'
            year_col = 'Year'
            # preview the uploaded file
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_table(uploaded_file, sep='\t')  # Doc: assume contents are seperated by Tabs.
            abstract_col = 'AB'
            title_col = 'TI'
            source_title_col = 'SO'
            year_col = 'PY'
            # preview the uploaded file
        else:
            st.error("Unsupported file format.")
            st.stop()

        column_choices = (abstract_col, title_col)

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            choice = st.selectbox("Choose column to analyze", column_choices)
        with row1_col2:
            comparison_options = ('Sources', 'Years')
            type_of_comparison = st.selectbox("Type of comparison", comparison_options)
        if type_of_comparison == 'Sources':
            compatison1(choice)
        if type_of_comparison == 'Years':
            comparison2(choice)


