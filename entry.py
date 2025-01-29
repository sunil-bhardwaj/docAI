import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Updated imports
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_community.llms import Ollama
from langchain_community.llms.ollama import Ollama
from langchain.chains import create_retrieval_chain
import pickle
import time
from rank_bm25 import BM25Okapi
###########################################################################

###########################################################################
start = time.time()
st.set_page_config(page_title="PDFChat")
st.title("**PdfChat: An AI PDF Analysis Tool**")
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_METADATA_PATH = "faiss_metadata.pkl"
LAST_PDF_PATH = "last_uploaded_pdf.txt"
LAST_MODEL_PATH = "last_used_model.txt"
bm25 = None
bm25_docs = None
"""
Model	Speed	Size	         Best For
Mistral 7B	🚀 Fast	            🟢 Small	    Semantic Search
Gemma 2B	⚡ Super Fast	   🟢 Tiny	       Lightweight Q&A
Llama 3 8B	⚡ Medium	       🔵 Medium	   Strong Reasoning
Phi-3 Mini	🚀 Fast	            🟢 Small	    Code & Docs
TinyLlama	⚡ Super Fast	   🟢 1.1B	       Low-RAM Devices
Falcon 7B	🚀 Fast	            🔵 Medium	    Conversational AI
Zephyr 7B	⚡ Fast	           🔵 Medium	   Chatbots & Q&A

"""

available_models = ["llama2", "llama3", "llama3.3", "gemma", "mistral", "deepseek-r1", "tinyllama","phi4","Falcon","Zephyr"] # Search for some more efficient models free and open source
selected_model = st.selectbox("Choose an AI Model:", available_models)
# Initialize LLM based on selected model
model_choice = st.selectbox("Choose the LLM Provider:", ["Ollama", "GPT4All", "Mistral", "Gemma", "TinyLlama", "Zephyr"])
if model_choice == "Ollama":
    llm = OllamaLLM(model=selected_model)
    embeddings = OllamaEmbeddings(model=selected_model)
elif model_choice == "GPT4All":
    llm = GPT4All(model=selected_model)  # You can customize with specific models for GPT4All.
    embeddings = OllamaEmbeddings(model=selected_model) 
else:
    # Implement other LLM options if needed, such as Mistral, Gemma, etc.
    llm = OllamaLLM(model=selected_model)  # Default to Ollama for other models
    embeddings = OllamaEmbeddings(model=selected_model)

st.text(f"Created OllamaLLM ({selected_model}) in  : {time.time() - start:.2f} seconds")

###########################################################################
# ----------------------------------------
# 📂 Allow User to Upload a PDF File
# ----------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is None:
    st.warning("Please upload a PDF to proceed.")
    st.stop()
# Save uploaded PDF to a temporary file
PDF_FILE = "uploaded.pdf"
with open(PDF_FILE, "wb") as f:
    f.write(uploaded_file.getbuffer()) 
# ----------------------------------------
# 🔄 Check if this is a new PDF
# ----------------------------------------
model_changed, new_pdf_uploaded = True, True
if os.path.exists(LAST_MODEL_PATH):
    with open(LAST_MODEL_PATH, "r") as f:
        last_model = f.read().strip()
    if last_model == selected_model:
        model_changed = False  # Model is the same
if os.path.exists(LAST_PDF_PATH):
    with open(LAST_PDF_PATH, "r") as f:
        last_pdf_name = f.read().strip()
    if last_pdf_name == uploaded_file.name:
        new_pdf_uploaded = False  # Same PDF
###########################################################################
def tokenize(text):
    return text.lower().split()
def create_bm25_index(documents):
    global bm25, bm25_docs
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_docs = documents


###########################################################################


st.text(f"Created OllamaLLM in  : {time.time() - start:.2f} seconds")
embeddings = OllamaEmbeddings(model=selected_model)
st.text(f"Created Embedding ({selected_model}) in  : {time.time() - start:.2f} seconds")
###########################################################################
# Function to Save FAISS Vectorstore#########################################
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
##############################################################################
# Function to Load FAISS Vectorstore##########################################
def load_faiss_vectorstore():
    """Loads FAISS vectorstore if available."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        st.text("Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        docstore = InMemoryDocstore(metadata["documents"])
        index_to_docstore_id = metadata.get("index_to_docstore_id", {})
        max_index = index.ntotal
        index_to_docstore_id = {i: index_to_docstore_id.get(i, str(i)) for i in range(max_index)}
        st.text(f"Time taken Till load EXISTING faiss_vectorstore : {time.time() - start:.2f} seconds")
        documents = list(docstore._dict.values())  # Extract stored documents
        
        create_bm25_index(documents)  
        return FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=embeddings.embed_query  # Required for FAISS
        )
    return None
##############################################################################
if new_pdf_uploaded:
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
    create_bm25_index(docs)
    # Save new FAISS index
    save_faiss_vectorstore(db)
    # Store the uploaded filename
    with open(LAST_PDF_PATH, "w") as f:
        f.write(uploaded_file.name)
else:
    st.text("🔄 Using previously loaded FAISS index...")
    db = load_faiss_vectorstore()
    


##############################################################################
# if db is None:
#     # If not available, process the PDF and create it
#     st.text("Processing PDF and creating FAISS index...")
    
#     loader = PyPDFLoader(PDF_FILE)
#     docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
    
#     db = FAISS.from_documents(docs, embeddings)
#     st.text(f"Time taken to create NEW faiss_vectorstore : {time.time() - start:.2f} seconds")
#     # Save FAISS index for future use
#     save_faiss_vectorstore(db)
##############################################################################
st.text("FAISS vectorstore is ready.")

st.text("Creating ChatPromptTemplate...")
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
st.text("Creating document chain...")
document_chain = create_stuff_documents_chain(llm,prompt)
st.text("Creating retriever...")
retirever = db.as_retriever(search_kwargs={"k": 5})


#################################### **Query Expansion Function**
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

# ----------------------------------------
#################################### **Fix: Handle missing documents gracefully**
def bm25_search(query, top_k=5):
    if bm25 is None or bm25_docs is None:
        raise ValueError("BM25 index is not initialized!")
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_docs[i] for i in top_n_indices]
def hybrid_search(query, retirever):
    """Combine BM25 and FAISS results for improved retrieval."""
    faiss_results = retirever.get_relevant_documents(query)
    bm25_results = bm25_search(query)

    # Merge & remove duplicates
    all_results = {doc.page_content: doc for doc in faiss_results + bm25_results}.values()
    return list(all_results)
def safe_similarity_search1(query):
    try:
        return retirever.get_relevant_documents(query)
    except ValueError as e:
        st.error(f"Error: {e}")
        st.text("Rebuilding FAISS index...")
        os.remove(FAISS_INDEX_PATH)
        os.remove(FAISS_METADATA_PATH)
        st.text("Please restart the app.")
        return []
 ################################################# 
def safe_similarity_search2(query):
    """Performs query expansion and retrieves relevant documents."""
    try:
        # Expand user query
        expanded_queries = expand_query(query)
        
        # Retrieve documents for all expanded queries
        all_docs = []
        for expanded_query in expanded_queries:
            all_docs.extend(retirever.get_relevant_documents(expanded_query))
        
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
################################################# 
st.text("Creating retrieval chain...")
retrieval_chain = create_retrieval_chain(retirever,document_chain)
st.text(f"Time taken Till Type your query : {time.time() - start:.2f} seconds")
search_method = st.selectbox("Choose Search Method:", 
                             ("BM25 (Fast Keyword Search)", 
                              "FAISS Embeddings (Semantic Search)", 
                              "Hybrid Search (BM25 + FAISS)","Basic Search","Semantic Search"))
prompt = st.text_input("Type your query")


def selected_similarity_search(query):
    if search_method == "Basic Search":
        return safe_similarity_search1(query)
    elif search_method == "Semantic Search":
        return safe_similarity_search2(query)
    elif search_method == "BM25 (Fast Keyword Search)":
        return bm25_search(query)
    elif search_method == "FAISS Embeddings (Semantic Search)":
        return retirever.get_relevant_documents(query)
    elif search_method == "Hybrid Search (BM25 + FAISS)":
        return hybrid_search(query, retirever)
    
if prompt:
    st.text("Processing query...")
    try:
        documents = selected_similarity_search(prompt)
        if not documents:
            st.text("No relevant documents found.")
        else:
            response = retrieval_chain.invoke({"input": prompt})
            st.text("Query completed.")
            print("Answer:", response['answer'])
            st.write(response['answer'])
            st.text(f"Total Time taken: {time.time() - start:.2f} seconds")
    except KeyError as e:
        st.text(f"Error: Missing document ID {e}. Try rebuilding the FAISS index.")

    

   
    
