# ===================================================================================
# Project: GitSurfer
# File: app/graphs/git_assistant.py
# Description: This file contains the implementation of Git Assistant Graph.
# Author: LALAN KUMAR
# Created: [20-05-2025]
# Updated: [22-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
import re
from typing import Any, Dict, List, Optional, cast, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt, Command
import asyncio

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from logger import logging
from app.core.llm import get_llm
from app.graphs.fetcher import create_fetcher_graph
from app.graphs.embedder import create_embedder_graph
from app.graphs.researcher import create_researcher_graph
from app.graphs.states import FetchState, EmbedderState, AgentState
from app.graphs.prompts import RESPONSE_SYSTEM_PROMPT, RESEARCH_PLAN_SYSTEM_PROMPT
from app.core.utils import format_docs

# Node: Parse GitHub URL (async allowed here)
async def parse_github_url(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    if not state.github_url:
        return {"error": "No GitHub URL provided"}

    patterns = [
        r"https://github\.com/([^/]+)/([^/]+)/?(?:/tree/([^/]+))?",
        r"github\.com/([^/]+)/([^/]+)/?(?:/tree/([^/]+))?",
        r"([^/]+)/([^/]+)/?(?:/tree/([^/]+))?"
    ]

    for pattern in patterns:
        match = re.match(pattern, state.github_url.strip())
        if match:
            owner = match.group(1)
            repo = match.group(2).rstrip('.git')
            branch = match.group(3) if match.group(3) else "main"

            logging.info(f"Parsed GitHub URL: owner={owner}, repo={repo}, branch={branch}")
            return {
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "error": None
            }

    return {"error": f"Invalid GitHub URL format: {state.github_url}"}

# Node: Fetch Repository (async allowed)
async def fetch_repository(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    try:
        logging.info(f"Fetching repository: {state.owner}/{state.repo}")

        fetcher_graph = create_fetcher_graph()
        fetch_state = FetchState(
            owner=state.owner,
            repo=state.repo,
            branch=state.branch
        )

        result = await fetcher_graph.ainvoke(fetch_state, config=config)

        files = result.get("files")
        if files is not None:
            logging.info(f"Repository fetched successfully: {len(files)} files")
            return {
                "repo_fetched": True,
                "error": None
            }
        else:
            logging.error("Fetcher graph did not return any files.")
            return {
                "repo_fetched": False,
                "error": "Fetcher graph did not return any files."
            }

    except Exception as e:
        logging.error(f"Error fetching repository: {e}")
        return {
            "repo_fetched": False,
            "error": f"Failed to fetch repository: {str(e)}"
        }

# Node: Create Vector Database (async allowed)
async def create_vectordb(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    try:
        logging.info("Creating vector database...")

        embedder_graph = create_embedder_graph()
        embedder_state = EmbedderState()

        result = await embedder_graph.ainvoke(embedder_state, config=config)
        logging.info(f"Embedder graph completed. Final state: {result}")

        if result.get("retriever_ready") and result.get("retriever"):
            logging.info("Vector database created successfully")
            return {
                "vectordb_ready": True,
                "retriever": result["retriever"],
                "waiting_for_query": True,
                "error": None
            }
        else:
            return {
                "vectordb_ready": False,
                "error": result.get("error") or "Failed to create vector database"
            }

    except Exception as e:
        logging.error(f"Error creating vector database: {e}")
        return {
            "vectordb_ready": False,
            "error": f"Failed to create vector database: {str(e)}"
        }

# Node: Wait for User Query (synchronous, calls interrupt)
async def collect_user_query(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    logging.info(f"[collect_user_query] user_query: {state.user_query}, messages: {[m.content for m in state.messages]}")
    if state.user_query:
        messages = state.messages + [HumanMessage(content=state.user_query)]
        return {
            "messages": messages,
            "waiting_for_query": False,
            "user_query": None
        }

    prompt = "Repository has been processed and is ready for questions! What would you like to know about this codebase?"
    user_input = await asyncio.to_thread(input, prompt)  # Local CLI input
    messages = state.messages + [HumanMessage(content=user_input)]
    return {
        "messages": messages,
        "waiting_for_query": False,
        "user_query": None
    }

# Node: Create Research Plan (async allowed)
async def create_research_plan(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    class Plan(TypedDict):
        steps: List[str]
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + [{"role": msg.type, "content": msg.content} for msg in state.messages]
    logging.info(f"[create_research_plan] messages: {[m['content'] for m in messages]}")

    llm = get_llm(
        streaming=config.get("streaming", False) if config else False,
        callbacks=config.get("callbacks", []) if config else []
    )
    model = llm.with_structured_output(Plan)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + [{"role": msg.type, "content": msg.content} for msg in state.messages]

    response = cast(Plan, await model.ainvoke(messages, config=config))
    logging.info(f"[create_research_plan] steps generated: {response['steps']}")
    logging.info(f"Created research plan with {len(response['steps'])} steps")
    return {
        "steps": response["steps"],
        "documents": []
    }

# Node: Conduct Research (async allowed)
async def conduct_research(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    logging.info(f"[conduct_research] steps: {state.steps}")
    if not state.steps:
        return {"documents": []}

    try:
        researcher_graph = create_researcher_graph()
        result = await researcher_graph.ainvoke(
            {"question": state.steps[0]},
            config=config
        )

        logging.info(f"[conduct_research] documents found: {len(result.get('documents', []))}")
        logging.info(f"Research conducted: found {len(result.get('documents', []))} documents")
        return {
            "documents": result.get("documents", []),
            "steps": state.steps[1:]
        }

    except Exception as e:
        logging.error(f"Error during research: {e}")
        return {
            "documents": [],
            "steps": state.steps[1:]
        }

# Node: Respond to User (async allowed)
async def respond(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    llm = get_llm(
        streaming=config.get("streaming", False) if config else False,
        callbacks=config.get("callbacks", []) if config else []
    )

    context = format_docs(state.documents)
    logging.info(f"[respond] context: {context[:200]}")  # Truncate for readability
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)

    messages = [
        {"role": "system", "content": prompt + "\n\nIMPORTANT: Always preserve code blocks with ```python and ``` markers. Never modify code content."}
    ] + [{"role": msg.type, "content": msg.content} for msg in state.messages]
    logging.info(f"[respond] messages: {[m['content'] for m in messages]}")
    response = await llm.ainvoke(messages, config=config)
    logging.info(f"[respond] response: {response.content}")
    print(response.content)
    follow_up_content = response.content + "\n\nDo you have any more questions about this codebase? (yes/no)"
    follow_up_message = AIMessage(content=follow_up_content)
    return {"messages": [follow_up_message]}

# Node: Check for More Questions (synchronous, calls interrupt)
def check_continue(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    logging.info(f"[check_continue] continue_conversation: {state.continue_conversation}")
    if state.continue_conversation is not None:
        if state.continue_conversation is True:
            return {
                "waiting_for_query": True,
                "continue_conversation": None,
                "documents": []
            }
        else:
            return {
                "waiting_for_query": False,
                "continue_conversation": None
            }

    prompt = "Do you have more questions about this codebase? Please answer yes or no."
    user_input = input(prompt)  # Local CLI input
    normalized = user_input.strip().lower()
    if normalized in ("yes", "y"):
        continue_conv = True
    elif normalized in ("no", "n"):
        continue_conv = False
    else:
        # Invalid input, interrupt again
        return check_continue(state, config=config)

    return {
        "continue_conversation": continue_conv
    }

# Conditional functions (unchanged)
def check_url_parsed(state: AgentState) -> Literal["fetch_repository", "end_with_error"]:
    if state.error:
        return "end_with_error"
    return "fetch_repository"

def check_repo_fetched(state: AgentState) -> Literal["create_vectordb", "end_with_error"]:
    if state.error or not state.repo_fetched:
        return "end_with_error"
    return "create_vectordb"

def check_vectordb_ready(state: AgentState) -> Literal["collect_user_query", "end_with_error"]:
    if state.error or not state.vectordb_ready:
        return "end_with_error"
    return "collect_user_query"

def check_query_collected(state: AgentState) -> Literal["create_research_plan", "collect_user_query"]:
    if state.waiting_for_query:
        return "collect_user_query"
    return "create_research_plan"

def check_research_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"

def check_conversation_continue(state: AgentState) -> Literal["collect_user_query", "end_conversation", "check_continue"]:
    if state.continue_conversation is True:
        return "collect_user_query"
    elif state.continue_conversation is False:
        return "end_conversation"
    else:
        return "check_continue"

# End nodes
async def end_with_error(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    error_message = AIMessage(content=f"Sorry, I encountered an error: {state.error}")
    return {"messages": [error_message]}

async def end_conversation(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    goodbye_message = AIMessage(content="Thank you for using the Git Assistant! Feel free to ask about other repositories anytime.")
    return {"messages": [goodbye_message]}

def create_git_assistant_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_github_url", parse_github_url)
    builder.add_node("fetch_repository", fetch_repository)
    builder.add_node("create_vectordb", create_vectordb)
    builder.add_node("collect_user_query", collect_user_query)
    builder.add_node("create_research_plan", create_research_plan)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("respond", respond)
    builder.add_node("check_continue", check_continue)
    builder.add_node("end_with_error", end_with_error)
    builder.add_node("end_conversation", end_conversation)

    builder.add_edge(START, "parse_github_url")

    builder.add_conditional_edges(
        "parse_github_url",
        check_url_parsed,
        {
            "fetch_repository": "fetch_repository",
            "end_with_error": "end_with_error"
        }
    )

    builder.add_conditional_edges(
        "fetch_repository",
        check_repo_fetched,
        {
            "create_vectordb": "create_vectordb",
            "end_with_error": "end_with_error"
        }
    )

    builder.add_conditional_edges(
        "create_vectordb",
        check_vectordb_ready,
        {
            "collect_user_query": "collect_user_query",
            "end_with_error": "end_with_error"
        }
    )

    builder.add_conditional_edges(
        "collect_user_query",
        check_query_collected,
        {
            "create_research_plan": "create_research_plan",
            #"collect_user_query": "collect_user_query"
        }
    )

    builder.add_edge("create_research_plan", "conduct_research")

    builder.add_conditional_edges(
        "conduct_research",
        check_research_finished,
        {
            "respond": "respond",
            "conduct_research": "conduct_research"
        }
    )

    builder.add_edge("respond", "check_continue")

    builder.add_conditional_edges(
        "check_continue",
        check_conversation_continue,
        {
            "collect_user_query": "collect_user_query",
            "end_conversation": "end_conversation",
            "check_continue": "check_continue"
        }
    )

    builder.add_edge("end_with_error", END)
    builder.add_edge("end_conversation", END)

    git_assistant_graph = builder.compile()
    git_assistant_graph.name = "GitAssistantGraph"

    return git_assistant_graph

# Interactive example usage
async def interactive_main():
    graph = create_git_assistant_graph()

    initial_state = AgentState(
        github_url="https://github.com/kumar8074/HyperSpectral-AI"
    )

    config = {"thread_id": "conversation-1"}

    print("🔄 Processing repository...")
    result = await graph.ainvoke(initial_state, config=config)
    logging.info(f"Repository processed. Final state: {result}")

    while True:
        # Handle interrupt
        if "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0]
            prompt = interrupt_obj.value if hasattr(interrupt_obj, "value") else "Input required:"
            user_input = input(f"👤 {prompt} ").strip()
            result = await graph.ainvoke(Command(resume=user_input), config=config)
            # Print messages (always print after ainvoke)
            if "messages" in result:
                for message in result["messages"]:
                    print(f"🤖 Assistant: {message.content}")
                print("-" * 50)
            continue

        # Print messages (always print after ainvoke)
        if "messages" in result:
            for message in result["messages"]:
                print(f"🤖 Assistant: {message.content}")
            print("-" * 50)

        # Check if waiting for user query flag (fallback)
        if result.get("waiting_for_query", False):
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue
            updated_state = AgentState(**dict(result))
            updated_state.user_query = user_input
            result = await graph.ainvoke(updated_state, config=config)
            # Print assistant response immediately after user query
            if "messages" in result:
                for message in result["messages"]:
                    print(f"🤖 Assistant: {message.content}")
                print("-" * 50)
            continue

        # Check if waiting for continue_conversation flag
        if "continue_conversation" in result and result["continue_conversation"] is None:
            user_response = input("👤 Continue? (yes/no): ").strip().lower()
            if user_response in ['yes', 'y']:
                updated_state = AgentState(**dict(result))
                updated_state.continue_conversation = True
                result = await graph.ainvoke(updated_state, config=config)
            elif user_response in ['no', 'n']:
                updated_state = AgentState(**dict(result))
                updated_state.continue_conversation = False
                result = await graph.ainvoke(updated_state, config=config)
                # Print assistant response immediately after user says no
                if "messages" in result:
                    for message in result["messages"]:
                        print(f"🤖 Assistant: {message.content}")
                    print("-" * 50)
                break
            else:
                print("Please answer 'yes' or 'no'")
            continue

        # No interrupts or waiting flags, end conversation
        break


# Main entrypoint
async def main():
    await interactive_main()

if __name__ == "__main__":
    asyncio.run(main())


