import streamlit as st
import os
import shutil

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from core.ingestion import load_all_documents, get_hierarchical_nodes
from core.retrieval import setup_index_and_retriever, get_reranker
from core.generation import create_query_engine

st.set_page_config(page_title="AI Research Copilot", layout="wide")

@st.cache_resource
def initialize_ai_models():
    """Loads the massive AI models only ONCE into Mac memory."""
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    
    Settings.llm = Ollama(
        model="gemma3:1b", 
        request_timeout=360.0, 
        context_window=4096,  
        temperature=0.3,       
        additional_kwargs={"num_predict": 1500} 
    )
    
    return get_reranker(), Settings.llm

reranker, llm = initialize_ai_models()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None


st.title("AI Research Copilot")
st.markdown("Upload Research Papers, Data, or Code Repos, then ask questions. Powered by **gemma3:1b**.")

with st.sidebar:
    st.header("1. Ingest Data")
    
    uploaded_files = st.file_uploader(
        "Upload Files (Images automatically skipped)", 
        accept_multiple_files=True
    )
    
    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Parsing, Chunking, and Embedding..."):
                temp_dir = "./temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                docs = load_all_documents(temp_dir)
                
                if len(docs) == 0:
                    st.error("No valid text/code files found.")
                else:
                    leaf_nodes, _ = get_hierarchical_nodes(docs)
                    hybrid_retriever = setup_index_and_retriever(leaf_nodes)
                    
                    st.session_state.query_engine = create_query_engine(hybrid_retriever, reranker, llm)
                    st.success(f"Successfully indexed valid files!")
                    
                shutil.rmtree(temp_dir)
        else:
            st.error("Please upload at least one file first.")

st.header("2. Ask Questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What is the main finding in the dataset?"):
    if st.session_state.query_engine is None:
        st.warning("Please upload and index a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            streaming_response = st.session_state.query_engine.query(prompt)
            full_response = st.write_stream(streaming_response.response_gen)
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})