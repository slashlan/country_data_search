### ECON DATA SOURCES SEARCH ENGINE ###
# to launch the app on ANACONDA console, activate virtual env and run: streamlit run C:\Users\pistis\PycharmProjects\econ_search\main.py
# to generate requirements.txt run this on command prompt console: pipreqs C:\Users\pistis\PycharmProjects\econ_search

# IMPORT LIBRARIES

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import base64


# Define preprocessing function
# @st.cache
def preprocess_text(dataframe):
    """Puts together text fields for each instance, cleans them
    and makes them ready for semantic similarity model. Also, it creates labels for each instance."""

    # Create list of texts to feed SBERT model
    sentence_list = dataframe['Description'].tolist()

    # Export entries titles and links as series
    instances_series = pd.Series(dataframe['Source name'])
    instances_links = pd.Series(dataframe['Link'])

    # Calculate text embeddings for entries in the database
    embeddings = model.encode(sentence_list)

    return sentence_list, instances_series, instances_links, embeddings


# Define similarity function to propose most similar suggestions by semantic similarity

def similarity_table(new_entry, instances_series, instances_links, embeddings):
    """Computes text embeddings for new entry and all entries already in the database.
    Calculates the cosine similarity vector and shows the most similar (cosine similarity > 0.2)
    database entries to the new entry."""

    # Encode text new entry
    new_embed = model.encode(new_entry)

    # Compute cosine similarity between new text & database
    cos_sim = util.cos_sim(new_embed, embeddings)

    # Put cos_sim in a Dataframe with labels and links
    similarity_vector_values = pd.DataFrame(cos_sim.numpy()).squeeze(axis=0)

    # Create table with cosine similarity, entries titles and links
    similarity_df = pd.concat([similarity_vector_values, instances_series, instances_links], axis=1).rename(
        columns={0: 'similarity_vector_values'})

    # Sort by higher similarity score and show top 10
    result = similarity_df.sort_values('similarity_vector_values', ascending=False).loc[
        similarity_df['similarity_vector_values'] > 0.2]

    return result


# APP START

# image_title = Image.open('C:/Users/pistis/PycharmProjects/econ_search/econ_search_logo_main_page_3.png')
# st.image(image_title, width=200)
# st.title('Country Source Finder')

file_ = open("C:/Users/pistis/PycharmProjects/econ_search/Country source finder cropped.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)
st.caption(
    'Country Data Source Finder is a search engine that helps you finding the best data sources for the country-level statistics '
    'that you seek. Type what you are looking for in the search bar and get the right data source you need.')

with st.spinner(text='Loading..'):
    # Define semantic model to use
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('all-distilroberta-v1')
    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Load data and embeddings
    df = pd.read_excel('C:/Users/pistis/PycharmProjects/econ_search/Economic_data_sources_2023.xlsx')

    # Preprocess data for app
    sentence_list, title_list, link_list, embeddings = preprocess_text(df)
# st.success('Done!')

# SEARCH TAB
# Set up columns main page
col1, col2 = st.columns(2)

with col1:
    value = st.text_input('', max_chars=128, placeholder='Search..')
with col2:
    st.write('')
    st.write('')
    search_button = st.button('Find')

# Search results
if not value and not search_button:
    print('')
else:
    # Create similarity vector and top 3 most similar scores
    results_search = similarity_table(value, title_list, link_list, embeddings)

    # Print results
    for i in range(0, len(results_search)):
        result_box = st.container()
        label_source = str(results_search['Source name'].iloc[i])
        link_source = str(results_search['Link'].iloc[i])
        result_box.subheader("🗃️ [{source}]({link})".format(source=label_source, link=link_source))
