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