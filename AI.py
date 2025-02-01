import streamlit as st
import time
import os
import faiss
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Updated imports
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize start time and Streamlit configuration
start = time.time()
st.set_page_config(page_title="PDFChat")
st.title("**HC HIGH COURT AI: An AI Chat Bot, PDF Analysis Tool**")

FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_METADATA_PATH = "faiss_metadata.pkl"
LAST_PDF_PATH = "last_uploaded_pdf.txt"

# Available models for user to choose
available_models = [
    "llama2", "llama3", "llama3.3", "gemma", "mistral", "deepseek-r1", "tinyllama",
    "phi4", "Falcon", "Zephyr", "GPT-J--NONE", "GPT-NeoX--NONE", "BLOOM--NONE", "OPT--NONE", "T5--NONE", "Alpaca--NONE", "BERT--NONE", "RoBERTa--NONE", "DistilGPT2--NONE"
]

# Let the user select a model
selected_model = st.selectbox("Select LLM Model:", available_models)

# 📂 Allow User to Upload a PDF File
uploaded_file = st.file_uploader("Upload a PDF (optional)", type=["pdf"])

# Check if file is uploaded and handle accordingly
if uploaded_file:
    # Save uploaded PDF to a temporary file
    PDF_FILE = "uploaded.pdf"
    with open(PDF_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Check if this is a new PDF
    new_pdf_uploaded = True
    if os.path.exists(LAST_PDF_PATH):
        with open(LAST_PDF_PATH, "r") as f:
            last_pdf_name = f.read().strip()
        if last_pdf_name == uploaded_file.name:
            new_pdf_uploaded = False  # Same file, no need to rebuild
else:
    st.warning("⚠️ Please upload a PDF to proceed (optional).")
    new_pdf_uploaded = False

# Initialize the selected LLM and embeddings based on user selection
if selected_model == "llama2":
    llm = OllamaLLM(model="llama2")
    embeddings = OllamaEmbeddings(model="llama2")
elif selected_model == "llama3":
    llm = OllamaLLM(model="llama3")
    embeddings = OllamaEmbeddings(model="llama3")
elif selected_model == "llama3.3":
    llm = OllamaLLM(model="llama3.3")
    embeddings = OllamaEmbeddings(model="llama3.3")
else:
    llm = OllamaLLM(model=selected_model)
    embeddings = OllamaEmbeddings(model=selected_model)

st.text(f"Created {selected_model} in  : {time.time() - start:.2f} seconds")

# Function to Save FAISS Vectorstore
def save_faiss_vectorstore(vectorstore):
    """Saves FAISS vectorstore index and metadata."""
    faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)
    metadata = {
            "documents": vectorstore.docstore._dict,  # Save document store
            "index_to_docstore_id": vectorstore.index_to_docstore_id  # Ensure correct key storage
        }
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    st.text(f"FAISS index saved in  : {time.time() - start:.2f} seconds")

# Function to Load FAISS Vectorstore
def load_faiss_vectorstore():
    """Loads FAISS vectorstore if available."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        st.text("Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        docstore = InMemoryDocstore(metadata["documents"])
        index_to_docstore_id = metadata.get("index_to_docstore_id", {})
        st.text(f"Time taken Till load EXISTING faiss_vectorstore : {time.time() - start:.2f} seconds")
        return FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=embeddings.embed_query  # Required for FAISS
        )
    return None

# Check if new PDF is uploaded and process it
if new_pdf_uploaded and uploaded_file:
    st.text("🆕 New PDF detected. Rebuilding FAISS index...")

    # Delete old FAISS index files
    for file in [FAISS_INDEX_PATH, FAISS_METADATA_PATH]:
        if os.path.exists(file):
            os.remove(file)
    # Process new PDF
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250))

    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"Page {doc.metadata.get('page', i)}"

    db = FAISS.from_documents(docs, embeddings)

    # Save new FAISS index
    save_faiss_vectorstore(db)
    # Store the uploaded filename
    with open(LAST_PDF_PATH, "w") as f:
        f.write(uploaded_file.name)
else:
    st.text("🔄 Using previously loaded FAISS index...")
    db = load_faiss_vectorstore()

# Creating ChatPromptTemplate for answering based on documents
st.text("Creating ChatPromptTemplate for document queries...")
prompt = ChatPromptTemplate.from_template(""" 
You are an AI assistant that answers questions based on the provided PDF context.

📌 **Instructions:**
- Answer only using the context provided.
- If unsure, say "I don't have enough information from the document."
- Keep answers **clear and concise**.

📝 **Context:**
{context}

❓ **User Question:**  
{input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)

# Creating retriever
st.text("Creating retriever for documents...")
retriever = db.as_retriever(search_kwargs={"k": 5})

# **Query Expansion Function**
def expand_query(query):
    """Expands query using LLM to generate relevant synonyms and variations."""
    expansion_prompt = f"""
    Generate alternative phrasings and synonyms for the following query:
    
    "{query}"
    
    Provide at least 3 alternative queries that mean the same thing.
    """
    expansion_result = llm.invoke(expansion_prompt)

    if expansion_result:
        expanded_queries = expansion_result.strip().split("\n")
        return [query] + expanded_queries  # Include original query
    else:
        return [query]

# **Safe Similarity Search**
def safe_similarity_search(query):
    """Performs query expansion and retrieves relevant documents."""
    try:
        # Expand user query
        expanded_queries = expand_query(query)

        # Retrieve documents for all expanded queries
        all_docs = []
        for expanded_query in expanded_queries:
            all_docs.extend(retriever.get_relevant_documents(expanded_query))

        # Remove duplicates
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        return list(unique_docs)

    except ValueError as e:
        st.error(f"⚠️ Error: {e}")
        st.text("Rebuilding FAISS index...")
        for file in [FAISS_INDEX_PATH, FAISS_METADATA_PATH]:
            if os.path.exists(file):
                os.remove(file)
        st.text("Please restart the app.")
        return []

# Create a retrieval chain
st.text("Creating retrieval chain for document queries...")
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Define a container for the query results
results_placeholder = st.empty()

# Define the input prompt at the bottom
prompt_input = st.text_input("Ask a question:", key="query_input")

# Function to display results word by word
def display_word_by_word(answer, speed=0.1):
    words = answer.split()
    display_text = ""
    for word in words:
        display_text += word + " "
        results_placeholder.markdown(display_text)  # Append word and update display
        time.sleep(speed)
    st.text(f"Time taken: {time.time() - start:.2f} seconds")

# Process the user query
if prompt_input:
    st.text("Processing query...")

    if uploaded_file:
        # If PDF is uploaded, process it with the document retrieval chain
        try:
            documents = safe_similarity_search(prompt_input)
            if not documents:
                results_placeholder.text("No relevant documents found in the PDF.")
            else:
                response = retrieval_chain.invoke({"input": prompt_input})
                display_word_by_word(response['answer'])
        except KeyError as e:
            results_placeholder.text(f"Error: Missing document ID {e}. Try rebuilding the FAISS index.")
    else:
        # If no PDF uploaded, answer using simple LLM
        try:
            response = llm.invoke(prompt_input)
            display_word_by_word(response)
        except Exception as e:
            st.error(f"⚠️ Error in processing LLM response: {e}")

# Add custom CSS to fix the prompt at the bottom and make results scrollable
st.markdown("""
    <style>
        /* Fix the input box at the bottom */
        div.stTextInput {
            position: static;
            bottom: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            z-index: 999;
            box-shadow: 0px -4px 10px rgba(0,0,0,0.1);
        }

        /* Make results container scrollable */
        .scrollable-container {
            max-height: calc(100vh - 100px); /* Prevents content from overflowing */
            overflow-y: auto;
            padding-bottom: 50px; /* Extra space to avoid content overlap */
        }

        /* Ensure results do not push input field down */
        .main {
            padding-bottom: 120px; /* Adds padding to avoid overlap of results with input */
        }
    </style>
""", unsafe_allow_html=True)
