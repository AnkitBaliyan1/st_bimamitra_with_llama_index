import streamlit as st
from fetch_query import *
from eval_resposne import *
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.title("version_2.0")

# Check if the index is not already set in the session state
if 'query_engine' not in st.session_state:
    # Perform the initial operation if index is not set
    with st.spinner("Getting things ready for you"):
        print("running prepare_index as index not in session state")
        query_engine = prepare_engine()
    # Set the index in the session state to avoid recomputation
    st.session_state.query_engine = query_engine
else:
    # Retrieve the index from the session state
    print("index fetched from session state")
    query_engine = st.session_state.query_engine

# st.write("App is now loaded")

# Take user input after the initial operation is done
user_query = st.text_area("Enter your query here...")

if st.button("Get Response"):
    try:
        # answer, elapsed_time = fetch_query_main(query_engine=query_engine, user_query=user_query)
        # st.write(answer)
        # st.success(f"\n\nTime taken to fetch response: {elapsed_time} seconds")

        response, elapsed_time, records= eval_response(user_query, query_engine, app_id = 'app')
        st.write(response.response)
        st.success(f"\n\nTime taken to fetch response using trulens: {elapsed_time} seconds")
        # st.dataframe(pd.DataFrame(records))
    except Exception as e:
        st.write("An error occurred while fetching response")
