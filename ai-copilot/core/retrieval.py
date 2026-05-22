import chromadb
from typing import List, Any
from llama_index.core import VectorStoreIndex, QueryBundle, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

class HybridRRFRetriever(BaseRetriever):
    """
    Highly accurate Hybrid Retriever using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(
        self, 
        vector_retriever: VectorIndexRetriever, 
        bm25_retriever: BM25Retriever,
        **kwargs: Any
    ) -> None:
        """Initialize the retrievers bypassing strict Pydantic validation."""
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        k = 60 
        fused_scores = {}
        nodes_dict = {}

        for rank, node in enumerate(vector_nodes):
            node_id = node.node.node_id
            nodes_dict[node_id] = node
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1 / (rank + k)

        for rank, node in enumerate(bm25_nodes):
            node_id = node.node.node_id
            nodes_dict[node_id] = node
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1 / (rank + k)

        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        final_nodes = []
        for node_id, score in reranked_results:
            node = nodes_dict[node_id]
            node.score = score 
            final_nodes.append(node)

        return final_nodes

def get_reranker():
    """Returns the reranker model so it can be cached in app.py"""
    return SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)

def setup_index_and_retriever(nodes):
    """Sets up the ChromaDB vector store and RRF retriever."""
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("ai_copilot_knowledge")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    
    return HybridRRFRetriever(
        vector_retriever=vector_retriever, 
        bm25_retriever=bm25_retriever
    )