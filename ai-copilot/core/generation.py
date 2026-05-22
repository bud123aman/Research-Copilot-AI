from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

CLEAN_PROMPT_TEMPLATE = (
    "You are an expert AI Research Copilot.\n"
    "Your goal is to answer the user's question purely based on the provided Context Information.\n"
    "Crucial Rules:\n"
    "1. Answer properly, accurately, and naturally.\n"
    "2. Provide proper definitions and highly detailed explanations if requested.\n"
    "3. The answer should be gramatically correct. \n"
    "---------------------\n"
    "Context Information:\n"
    "{context_str}\n"
    "---------------------\n"
    "User Question: {query_str}\n"
    "Answer:"
)
clean_prompt = PromptTemplate(CLEAN_PROMPT_TEMPLATE)

def create_query_engine(hybrid_retriever, reranker, llm):
    """Assembles the RAG pipeline with strict metadata filtering."""

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker], 
        llm=llm,
        text_qa_template=clean_prompt,
        streaming=True
    )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": clean_prompt}
    )
    
    return query_engine