import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

def load_all_documents(directory_path: str):
    """Loads documents, explicitly skipping media/images to prevent crashes."""
    media_files_to_exclude = [
        "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp", 
        "**/*.tiff", "**/*.gif", "**/*.bmp", "**/*.svg", "**/*.ico",
        "**/*.mp3", "**/*.wav", "**/*.mp4", "**/*.mov"
    ]
    
    reader = SimpleDirectoryReader(
        input_dir=directory_path, 
        recursive=True,
        exclude=media_files_to_exclude
    )
    return reader.load_data()

def get_hierarchical_nodes(documents):
    """Hierarchical chunking with overlap for perfect context preservation."""
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],
        chunk_overlap=20 
    )
    
    nodes = node_parser.get_nodes_from_documents(documents)
    return get_leaf_nodes(nodes), nodes