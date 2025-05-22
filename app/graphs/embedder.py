# ===================================================================================
# Project: GitSurfer
# File: app/graphs/embedder.py
# Description: This file contains the implementation of Embedder sub-Graph.
# Author: LALAN KUMAR
# Created: [20-05-2025]
# Updated: [22-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import asyncio
from langchain_core.runnables import RunnableConfig
from typing import Optional
from langgraph.graph import StateGraph, START, END

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from logger import logging
from config import settings
from app.retriever import data_ingestion
from app.retriever.retriever import get_retriever
from app.graphs.states import EmbedderState

# Paths
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../temp'))
CHUNKS_FILE = os.path.join(TEMP_DIR, 'chunks_raw.json')
TREE_FILE = os.path.join(TEMP_DIR, 'tree_summary.json')

# Node: check if files are ready
def check_files_node(state: EmbedderState, *, config: Optional[RunnableConfig] = None) -> EmbedderState:
    state = state.model_copy()
    state.files_ready = os.path.exists(CHUNKS_FILE) and os.path.exists(TREE_FILE)
    if not state.files_ready:
        state.error = "Required files not found."
        logging.error(f"Required files not found: chunks={os.path.exists(CHUNKS_FILE)}, tree={os.path.exists(TREE_FILE)}")
    else:
        state.error = None
        logging.info("Required files found and ready for processing")
    return state

# Async node: ingest data, create vector DB, and create retriever in one step
async def ingest_and_create_retriever_node(state: EmbedderState, *, config: Optional[RunnableConfig] = None) -> EmbedderState:
    state = state.model_copy()
    provider = state.provider or settings.EMBEDDING_PROVIDER
    
    try:
        logging.info(f"Starting data ingestion and retriever creation using provider: {provider}")
        
        # Call data_ingestion.main() which creates the vector DB and returns a retriever
        vector_db = await asyncio.to_thread(data_ingestion.main, provider)
        retriever = get_retriever()
        
        # Update state with successful results
        state.vector_db_created = True
        state.retriever_ready = True
        state.retriever = retriever
        state.provider = provider
        state.error = None
        
        logging.info(f"Vector DB and retriever created successfully using provider: {provider}")
        
    except Exception as e:
        # Update state with error results
        state.vector_db_created = False
        state.retriever_ready = False
        state.retriever = None
        state.error = str(e)
        logging.error(f"Error during data ingestion and retriever creation: {e}")
    
    return state

# Build the async LangGraph
def create_embedder_graph():
    builder = StateGraph(EmbedderState)
    builder.add_node("check_files", check_files_node)
    builder.add_node("ingest_and_create_retriever", ingest_and_create_retriever_node)

    # Define the flow
    builder.add_edge(START, "check_files")
    
    # Add conditional edge: only proceed to ingestion if files are ready
    def should_proceed(state: EmbedderState) -> str:
        if state.files_ready:
            return "ingest_and_create_retriever"
        else:
            return END
    
    builder.add_conditional_edges(
        "check_files",
        should_proceed,
        {
            "ingest_and_create_retriever": "ingest_and_create_retriever",
            END: END
        }
    )
    
    builder.add_edge("ingest_and_create_retriever", END)

    embedder_graph = builder.compile()
    embedder_graph.name = "EmbedderGraph"
    return embedder_graph

# Example usage:
async def async_main(provider=None):
    initial_state = EmbedderState(provider=provider)
    graph = create_embedder_graph()
    final_state = await graph.ainvoke(initial_state)
    logging.info(f"Embedder graph completed. Final state: {final_state}")
    return final_state

if __name__ == "__main__":
    asyncio.run(async_main())
