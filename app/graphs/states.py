# ===================================================================================
# Project: GitSurfer
# File: app/graphs/states.py
# Description: This file contains the state schemas used by the graphs.
# Author: LALAN KUMAR
# Created: [19-05-2025]
# Updated: [20-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from typing import List, Tuple, Dict, Any, Optional, Annotated
from operator import add
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
import os
import sys

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.utils import reduce_docs

# State used by the fetcher Graph
class FetchState(BaseModel):
    owner: str
    repo: str
    branch: str = "main"
    tree: List[Tuple[str, str]] = Field(default_factory=list)
    files: Annotated[List[Dict[str, Any]], add] = Field(default_factory=list)


# State used by the embedder Graph
class EmbedderState(BaseModel):
    files_ready: bool = False
    vector_db_created: bool = False
    retriever_ready: bool = False
    error: Optional[str] = None
    provider: Optional[str] = None
    retriever: Optional[Any] = None

# states used by the researcher graph
class QueryState(BaseModel):
    """Private state for the retrieve_documents node in the researcher graph"""
    query: str

class ResearcherState(BaseModel):
    """State of the researcher graph"""
    question: str
    queries: list[str] = Field(default_factory=list)
    documents: Annotated[list[Document], reduce_docs] = Field(default_factory=list)

# State for the main Git Assistant graph
class AgentState(BaseModel):
    """State for the Git Assistant agent"""
    # GitHub repository information
    github_url: Optional[str] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    branch: str = "main"
    
    # Processing states
    repo_fetched: bool = False
    vectordb_ready: bool = False
    retriever: Optional[Any] = None
    
    # Conversation state
    messages: List[BaseMessage] = Field(default_factory=list)
    user_query: Optional[str] = None
    
    # Research state
    steps: List[str] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    
    # Control flow
    waiting_for_query: bool = False
    continue_conversation: Optional[bool] = None
    
    # Error handling
    error: Optional[str] = None
