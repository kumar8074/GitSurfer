# GitSurfer

GitSurfer is an intelligent, multi-provider codebase analysis and research assistant for GitHub repositories. It leverages advanced LLMs (Gemini, OpenAI, Anthropic, Cohere) and vector databases to dynamically fetch, summarize, embed, and answer questions about any public GitHub repository, providing deep insights and research capabilities for developers and researchers.

---

## Features
- **Fetches and analyzes GitHub repositories** (tree structure, file contents)
- **Summarizes repository structure** using LLMs
- **Embeds code and documentation** into a vector database (ChromaDB)
- **Supports multiple LLM and embedding providers**: Gemini, OpenAI, Anthropic, Cohere
- **Interactive research assistant**: Ask questions about the codebase and get detailed, contextual answers
- **Extensible modular architecture** using LangGraph and LangChain
- **Rich logging and error handling**

---

## Project Structure
```
GitSurfer/
├── app/
│   ├── core/           # Core utilities, LLM/embedding logic
│   ├── graphs/         # Main assistant, fetcher, embedder, researcher graphs
│   ├── retriever/      # Data ingestion and retriever logic
├── config/             # Settings and environment variable loader
├── DATA/               # Persisted vector DBs
├── temp/               # Temporary files (chunks, summaries)
├── logs/               # Log files
├── logger.py           # Logging configuration
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-fork-or-repo-url>
   cd GitSurfer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Copy `.env.example` to `.env` and fill in your API keys:
     - `GOOGLE_API_KEY` (for Gemini)
     - `OPENAI_API_KEY` (for OpenAI)
     - `ANTHROPIC_API_KEY` (for Anthropic)
     - `COHERE_API_KEY` (for Cohere)
     - `GITHUB_TOKEN` (for increased GitHub API limits)
   - You can also specify model names and other settings in `.env`.

---

## Usage

The main entry point is the `app/graphs/git_assistant.py` script. It runs an interactive CLI assistant:

```bash
python app/graphs/git_assistant.py
```

**Workflow:**
1. Enter a GitHub repository URL when prompted.
2. GitSurfer fetches the repo, summarizes its structure, and creates a vector DB.
3. Ask any question about the codebase (design, functions, usage, etc.).
4. Interactively continue the research session or exit.

**Example:**
```
🔄 Processing repository...
👤 Input required: Enter GitHub repo URL
🤖 Assistant: Repository fetched and analyzed. Ask your question!
👤 You: What does the main.py file do?
🤖 Assistant: [detailed answer]
```

---

## Configuration
- All settings (provider selection, model names, directories) are managed in `config/settings.py` and via environment variables.
- Supports switching between providers for both LLM and embeddings.
- Vector DBs are persisted under `DATA/`.

---

## Prerequisites
- Python 3.9+
- API keys for at least one supported LLM/embedding provider
- (Optional) GitHub Personal Access Token for higher API rate limits

---

## Environment Variables
| Variable              | Description                         |
|-----------------------|-------------------------------------|
| GOOGLE_API_KEY        | Gemini API key                      |
| OPENAI_API_KEY        | OpenAI API key                      |
| ANTHROPIC_API_KEY     | Anthropic API key                   |
| COHERE_API_KEY        | Cohere API key                      |
| GITHUB_TOKEN          | GitHub token for API calls           |
| GEMINI_LLM_MODEL      | Gemini model name (default set)      |
| OPENAI_LLM_MODEL      | OpenAI model name (default set)      |
| ...                   | See `config/settings.py` for all     |

---

## Testing
Run tests using:
```bash
pytest
```

---

## Credits
- **Author:** Lalan Kumar ([kumar8074](https://github.com/kumar8074))
- Built with [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), and [ChromaDB](https://github.com/chroma-core/chroma)

---

## License
[MIT License](LICENSE)
