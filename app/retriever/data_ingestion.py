# ===================================================================================
# Project: GitSurfer
# File: app/retriever/data_ingestion.py
# Description: This file loads data from JSON files, stores it in a VectorStoreDB (Defaults to CHROMA),
#              and provides retriever functionality for the stored data.
# Author: LALAN KUMAR
# Created: [19-05-2025]
# Updated: [22-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import argparse
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)


from logger import logging
from config import settings
from app.core.embeddings import get_embeddings
from typing import Optional

def flatten_tree(tree, parent_path=""):
    """Recursively flatten the tree_summary.json into a path → metadata dict."""
    flat = {}
    for name, value in tree.items():
        path = os.path.join(parent_path, name) if parent_path else name
        if isinstance(value, dict):
            # Directory or file with metadata
            if "type" in value:
                flat[path] = value
            else:
                flat.update(flatten_tree(value, path))
        else:
            # File with no metadata
            flat[path] = {}
    return flat


def load_json_files(chunks_path, tree_path):
    """Load both chunks_raw.json and tree_summary.json (if present)."""
    # Load file chunks
    if not os.path.exists(chunks_path):
        logging.error(f"Chunks file not found: {chunks_path}")
        file_chunks = []
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            file_chunks = json.load(f)
        logging.info(f"Loaded {len(file_chunks)} file chunks...")

    # Load and flatten tree metadata
    if not os.path.exists(tree_path):
        logging.warning(f"tree_summary.json not found at: {tree_path}")
        tree_metadata = {}
    else:
        with open(tree_path, "r", encoding="utf-8") as f:
            tree = json.load(f)
        tree_metadata = flatten_tree(tree)
    return file_chunks, tree_metadata


def prepare_documents(file_chunks, tree_metadata=None):
    """Convert to LangChain Document objects, merging metadata from tree_summary.json if available."""
    documents = []
    for chunk in file_chunks:
        path = chunk.get("path", "")
        meta = {
            "source": path,
            "file_path": path,
            "file_name": os.path.basename(path),
            "file_type": os.path.splitext(path)[1][1:].lower(),
            "size": len(chunk.get("content", "")),
        }
        # Merge in metadata from tree_summary.json if available
        if tree_metadata and path in tree_metadata:
            meta.update(tree_metadata[path])
        documents.append(Document(page_content=chunk.get("content", ""), metadata=meta))
    logging.info(f"Prepared {len(documents)} documents for vectorization...")
    return documents


def split_documents(documents):
    """Split large documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def persist_vector_db(docs, embeddings, persist_directory):
    """Save documents to Chroma vector store."""
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb


def main(embedding_provider=None):
    """Main function to ingest data from temp/chunks_raw.json and tree_summary.json, and return a retriever.
    
    Args:
        embedding_provider: Optional provider name to use (defaults to settings.EMBEDDING_PROVIDER)
        
    Returns:
        A retriever instance ready to use for querying the vector store
    """
    # Use the specified provider or default from settings
    provider = embedding_provider or settings.EMBEDDING_PROVIDER
    logging.info(f"Using embedding provider: {provider}")

    # Load both JSON files (chunks and tree summary)
    chunks_path = os.path.join(settings.TEMP_DIR, "chunks_raw.json")
    tree_path = os.path.join(settings.TEMP_DIR, "tree_summary.json")
    file_chunks, tree_metadata = load_json_files(chunks_path, tree_path)

    # Convert to LangChain Document objects, merging metadata from tree_summary.json
    documents = prepare_documents(file_chunks, tree_metadata)

    # Split documents into chunks
    docs = split_documents(documents)
    logging.info(f"Split into {len(docs)} chunks...")

    # Get dynamic embedding model based on provider
    embeddings = get_embeddings(provider)
    logging.info(f"Using Embedding model: {embeddings}")

    # Set output directory based on provider name
    persist_directory = f"DATA/chroma_store_{provider.lower()}"
    os.makedirs(persist_directory, exist_ok=True)

    logging.info(f"Persisting vectorDB to {persist_directory}...(This might take a while)")
    vectordb = persist_vector_db(docs, embeddings, persist_directory)
    return vectordb

#main()
    

# def ingest_all_providers():
#     """Ingest data for all available providers."""
#     providers = [
#         settings.EMBEDDING_PROVIDER_GEMINI,
#         settings.EMBEDDING_PROVIDER_OPENAI,
#         settings.EMBEDDING_PROVIDER_COHERE
#     ]
    
#    for provider in providers:
#        # Check if API key exists for this provider
#        key_var_name = f"{provider.upper()}_API_KEY"
#        key = getattr(settings, key_var_name, None)
        
#        if key:
#            logging.info(f"Processing data for {provider} provider")
#            try:
#                vectordb = main(provider)
#                logging.info(f"Successfully processed data for {provider}")
#            except Exception as e:
#                logging.error(f"Error processing data for {provider}: {str(e)}")
        #else:
        #    logging.warning(f"Skipping {provider} - API key not found")


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Ingest data for vector stores")
#    parser.add_argument("--provider", type=str, 
#                        help="Specific embedding provider to use (gemini, openai, cohere)")
#    parser.add_argument("--all", action="store_true", 
#                        help="Process data for all providers with valid API keys")
    
    #args = parser.parse_args()
    
    #if args.all:
    #    ingest_all_providers()
    #elif args.provider:
        # Convert provider name to the constant from settings
#        provider_map = {
#            "gemini": settings.EMBEDDING_PROVIDER_GEMINI,
#            "openai": settings.EMBEDDING_PROVIDER_OPENAI,
#            "cohere": settings.EMBEDDING_PROVIDER_COHERE
#        }
        
        #provider = provider_map.get(args.provider.lower())
        #if provider:
        #    vectordb = main(provider)
        #else:
        #    logging.error(f"Unknown provider: {args.provider}")
    #else:
        # Use default provider from settings
        #vectordb = main()
        
# To ingest data for all providers with valid API keys:
#python processor/data_ingestion.py --all

# Or for a specific provider:
#python processor/data_ingestion.py --provider gemini
#python processor/data_ingestion.py --provider openai
#python processor/data_ingestion.py --provider cohere