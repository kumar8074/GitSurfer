# ===================================================================================
# Project: GitSurfer
# File: app/graphs/embedder.py
# Description: This file contains the implementation of Embedder sub-Graph.
# Author: LALAN KUMAR
# Created: [20-05-2025]
# Updated: [20-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import asyncio

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from logger import logging
from config import settings
from app.retriever import data_ingestion
from app.graphs.states import EmbedderState

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Paths
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../temp'))
CHUNKS_FILE = os.path.join(TEMP_DIR, 'chunks_raw.json')
TREE_FILE = os.path.join(TEMP_DIR, 'tree_summary.json')

# Removed custom provider resolution in favor of using the same approach as data_ingestion.py

# Async node: check if files are ready
def check_files_node(state: EmbedderState) -> EmbedderState:
    state = state.model_copy()
    state.files_ready = os.path.exists(CHUNKS_FILE) and os.path.exists(TREE_FILE)
    if not state.files_ready:
        state.error = "Required files not found."
    else:
        state.error = None
    return state

# Async node: ingest data and create vector DB
async def ingest_node(state: EmbedderState) -> EmbedderState:
    state = state.model_copy()
    # Use the same approach as data_ingestion.py
    provider = state.provider or settings.EMBEDDING_PROVIDER
    try:
        await asyncio.to_thread(data_ingestion.main, provider)
        state.vector_db_created = True
        state.error = None
        logging.info(f"Vector DB created successfully using provider: {provider}")
    except Exception as e:
        state.vector_db_created = False
        state.error = str(e)
        logging.error(f"Error during vector DB creation: {e}")
    return state

# Async node: create retriever interface
async def retriever_node(state: EmbedderState) -> EmbedderState:
    state = state.model_copy()
    # Use the same approach as data_ingestion.py
    provider = state.provider or settings.EMBEDDING_PROVIDER
    persist_dir = f"DATA/chroma_store_{provider.lower()}"
    try:
        # Use the new create_retriever function from data_ingestion
        await asyncio.to_thread(data_ingestion.create_retriever, persist_directory=persist_dir, embedding_provider=provider)
        state.retriever_ready = True
        state.error = None
        logging.info(f"Retriever interface created successfully using provider: {provider}")
    except Exception as e:
        state.retriever_ready = False
        state.error = str(e)
        logging.error(f"Error during retriever creation: {e}")
    return state

# Build the async LangGraph

def create_embedder_graph():
    builder = StateGraph(EmbedderState)
    builder.add_node("check_files", check_files_node)
    builder.add_node("ingest", ingest_node)
    builder.add_node("retriever", retriever_node)

    builder.add_edge(START, "check_files")
    builder.add_edge("check_files", "ingest")
    builder.add_edge("ingest", "retriever")
    builder.add_edge("retriever", END)

    embedder_graph = builder.compile()
    embedder_graph.name = "EmbedderGraph"
    return embedder_graph

# Async entry point
async def async_main(provider=None):
    initial_state = EmbedderState(provider=provider)
    graph = create_embedder_graph()
    final_state = await graph.ainvoke(initial_state)
    logging.info(f"Embedder graph completed. Final state: {final_state}")
    return final_state

if __name__ == "__main__":
    asyncio.run(async_main())
