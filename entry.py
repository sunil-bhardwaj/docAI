import os
import requests
import time
from utils import *
import streamlit as st
from streamlit_lottie import st_lottie
from transformers import pipeline
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.agents import AgentExecutor

# Change to use a text generation model compatible with CausalLM
nlp_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=100)  # Set max_new_tokens  # GPT-2 for text generation
hf_pipeline = HuggingFacePipeline(pipeline=nlp_pipeline)

# Define embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set page config
st.set_page_config(page_title="DocWise")
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation for loading
st_lottie(load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_G6Lxp3nm1p.json"), height=200, key='coding')

# Title of the app
st.title("**DocWise: An AI PDF Analysis Tool**")

# Session state initialization
if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False
    st.session_state['filename'] = None
    st.session_state['agent_executor'] = None
    st.session_state['store'] = None

# File upload and processing
if not st.session_state['uploaded']:
    st.write("Upload your **:blue[pdf]** and ask your personal AI assistant any questions about it!")
    input_file = st.file_uploader('Choose a file')

    if input_file and does_file_have_pdf_extension(input_file):
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        # Store PDF file
        path = store_pdf_file(input_file, upload_dir)
        scs = st.success("File successfully uploaded")
        filename = input_file.name

        with st.spinner("Analyzing document..."):
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            # Create vector store from the document pages
            store = Chroma.from_documents(pages, embeddings, collection_name="analysis")
            vectorstore_info = VectorStoreInfo(name=filename, description="Analyzing PDF", vectorstore=store)

            # Create the toolkit and agent executor with error handling
            toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=hf_pipeline)
            agent_executor = create_vectorstore_agent(
                llm=hf_pipeline,
                toolkit=toolkit,
                verbose=True,
                handle_parsing_errors=True
            )
         
            st.session_state['agent_executor'] = agent_executor
            st.session_state['store'] = store
        scs.empty()

        st.session_state['uploaded'] = True
        st.session_state['filename'] = filename

        st.rerun()

# Query handling after upload
if st.session_state['uploaded']:
    st.write(f"Enter your questions about the document '{st.session_state['filename']}' below:")
    prompt = "What document is all about"#st.text_input("\nType your query")
    if prompt:
        #prompt = prompt.strip('"')
        agent_executor = st.session_state['agent_executor']
        store = st.session_state['store']
        with st.spinner("Generating response..."):
            try:
                if agent_executor is None:
                    st.error("Agent executor is not properly initialized.")
                else:
                    # Run the agent with the input prompt
                    response = agent_executor.run(input="What is the main topic of the document?")

                    # Debugging: Check response type and content
                    st.write("Response with intermediate steps: ", response)
                    st.write("Type of raw agent response: ", type(response))
                    if response is None or response == "":
                        st.write("No response received from the agent.")
                    else:
                        st.write("Raw agent response: ", response)

                    # Check if the response contains an action or answer
                    if isinstance(response, dict):
                        if "answer" in response:
                            st.write("Answer: ", response["answer"])  # Display the answer if it's available
                        elif "action" in response:
                            st.write("Action: ", response["action"])  # Display the action if it's available
                        else:
                            st.write("Unable to generate a valid response.")
                    else:
                        st.write("Unexpected response format. Full response: ", response)

            except ValueError as e:
                st.error(f"A value parsing error occurred: {str(e)}")
                st.write("Error Value details: ", e)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Error details: ", e)

            # Perform similarity search and display results
            with st.expander("Similarity Search"):
                try:
                    search = store.similarity_search_with_score(prompt)
                    if search:
                        st.write(search[0][0].page_content)
                    else:
                        st.write("No results found in the document for the query.")
                except Exception as e:
                    st.error(f"An error occurred during similarity search: {str(e)}")
