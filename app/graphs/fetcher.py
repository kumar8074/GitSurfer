# ===================================================================================
# Project: GitSurfer
# File: app/graphs/fetcher.py
# Description: This file contains the implementation of Fetcher sub-Graph.
# Author: LALAN KUMAR
# Created: [19-05-2025]
# Updated: [19-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import aiohttp
import asyncio
import base64
import json
from dotenv import load_dotenv
load_dotenv()

from typing import Optional, List, Tuple, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from logger import logging
from app.graphs.states import FetchState
from config.settings import GITHUB_TOKEN, TEMP_DIR
from app.graphs.prompts import SUMMARIZE_STRUCTURE_SYSTEM_PROMPT
from app.core.llm import get_llm

# Auth headers
AUTH_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# --- Helper Functions ---
async def retry_async(func, retries=3, backoff_in_seconds=1, *args, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed with error: {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(backoff_in_seconds * 2 ** attempt)
            

async def fetch_github_tree(owner: str, repo: str, branch: str = "main") -> List[Tuple[str, str]]:
    """Fetch the repository tree from GitHub."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=AUTH_HEADERS) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Error fetching tree: {response.status} — {text}")
            data = await response.json()
            return [(item["path"], item["type"]) for item in data["tree"]]

# --- Node 1: Fetch Repository Tree ---
async def fetch_repository_tree(
    state: FetchState, *, config: Optional[RunnableConfig] = None
) -> dict[str, list[tuple[str, str]]]:
    """Fetch the repository tree from GitHub."""
    try:
        tree_data = await fetch_github_tree(state.owner, state.repo, getattr(state, "branch", "main"))
        logging.info(f"Fetched tree with {len(tree_data)} items")
        return {"tree": tree_data}
    except Exception as e:
        logging.error(f"Error fetching repository tree: {e}")
        raise

# --- Node 2: Summarize Tree Structure ---
async def summarize_tree_structure(
    state: FetchState, *, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Summarize the repository structure using an LLM."""
    # Convert the tree to a formatted string
    tree_text = "\n".join(f"- {path}" for path, typ in state.tree)
    max_length = 3000  # adjust based on token limits
    if len(tree_text) > max_length:
        tree_text = tree_text[:max_length] + "\n... (truncated)"
    
    # Get LLM instance with config
    llm = get_llm(
        streaming=config.get("streaming", False) if config else False,
        callbacks=config.get("callbacks", []) if config else []
    )
    
    # Prepare the prompt
    messages = [
        {"role": "system", "content": SUMMARIZE_STRUCTURE_SYSTEM_PROMPT},
        {"role": "user", "content": f"FILE TREE:\n{tree_text}"}
    ]
    
    # Get the LLM response
    response = await llm.ainvoke(messages, config=config or {})
    
    # Parse the response (assuming it's a JSON string)
    try:
        tree_summary = json.loads(response.content)
    except json.JSONDecodeError:
        logging.warning("LLM response is not valid JSON, saving raw content")
        tree_summary = response.content
    
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Save the tree summary
    with open(f"{TEMP_DIR}/tree_summary.json", "w", encoding="utf-8") as f:
        if isinstance(tree_summary, str):
            f.write(tree_summary)
        else:
            json.dump(tree_summary, f, indent=2)
    
    return {"tree_summary": tree_summary}

# --- Node 3: Fetch Repository Files ---
async def fetch_repository_files(
    state: FetchState, *, config: Optional[RunnableConfig] = None
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all files from the repository in parallel."""
    owner = state.owner
    repo = state.repo
    branch = getattr(state, "branch", "main")
    paths = [p for p, t in state.tree if t == "blob"]
    
    async def fetch_single_file(path: str) -> dict[str, Any]:
        """Fetch content of a single file with retry logic."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=AUTH_HEADERS) as resp:
                    if resp.status != 200:
                        logging.warning(f"Failed to fetch {path}: HTTP {resp.status}")
                        return {"path": path, "error": f"{resp.status}"}
                    data = await resp.json()
                    if data.get("encoding") == "base64":
                        content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                    else:
                        content = data.get("content", "")
                    return {"path": path, "content": content}
            except Exception as e:
                logging.error(f"Error fetching {path}: {e}")
                return {"path": path, "error": str(e)}
    
    # Process files in parallel with concurrency limit
    sem = asyncio.Semaphore(10)  # Limit concurrent requests
    async def limited_fetch(path: str) -> dict[str, Any]:
        async with sem:
            return await retry_async(fetch_single_file, path=path)
    
    results = await asyncio.gather(*(limited_fetch(p) for p in paths))
    
    # Filter out errors or empty content
    files = [res for res in results if res.get("content")]
    logging.info(f"Fetched {len(files)} files with content")
    
    # Save the results
    with open(f"{TEMP_DIR}/chunks_raw.json", "w", encoding="utf-8") as f:
        json.dump(files, f, indent=2)
    
    return {"files": files}


def create_fetcher_graph():
    """Create and return the fetcher graph for GitHub repository analysis."""
    # Initialize the graph with the FetchState
    builder = StateGraph(FetchState)
    
    # Add nodes to the graph
    builder.add_node("fetch_repository_tree", fetch_repository_tree)
    builder.add_node("summarize_tree_structure", summarize_tree_structure)
    builder.add_node("fetch_repository_files", fetch_repository_files)
    
    # Define the flow of the graph
    builder.add_edge(START, "fetch_repository_tree")
    builder.add_edge("fetch_repository_tree", "summarize_tree_structure")
    builder.add_edge("summarize_tree_structure", "fetch_repository_files")
    builder.add_edge("fetch_repository_files", END)
    
    # Compile the graph
    fetcher_graph = builder.compile()
    fetcher_graph.name = "FetcherGraph"
    
    return fetcher_graph

# Create the graph instance
fetcher_graph = create_fetcher_graph()


# Example usage:
initial_state = FetchState(
    owner="kumar8074", 
    repo="NOVA-AI", 
    branch="main"
)

logging.info("Starting repository analysis...")
result = asyncio.run(fetcher_graph.ainvoke(initial_state))
        
# The result will contain the final state with all the data
logging.info("Analysis completed successfully!")
logging.info(f"Total files processed: {len(result['files'])}")
        
# Access other data from the result
logging.info(f"Repository structure summary saved to {TEMP_DIR}/tree_summary.json")
logging.info(f"Raw file contents saved to {TEMP_DIR}/chunks_raw.json")
