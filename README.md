# langgraph-sandbox

Sandbox repo for experimenting with LangGraph + LangChain agents. 

## What is in this repo

This project contains a few standalone agent experiments:

- `agents/agent_bot.py`: minimal single-node LangGraph chat loop.
- `agents/memory_agent.py`: conversational loop with in-memory chat history and transcript export.
- `agents/ReAct_agent.py`: ReAct-style tool-using math agent (`add`, `subtract`, `multiply`).
- `agents/drafter_agent.py`: drafting assistant that updates and saves a text document via tools.
- `agents/rag_agent.py`: work-in-progress RAG prototype (currently incomplete and not runnable).
- `notes/langchain-message-attributes.ipynb`: notebook notes/scratch work.
- `main.py`: placeholder entrypoint.

## Prerequisites

- Python `3.11+`
- An OpenAI API key

## Setup

This repo is configured for `uv`.

```bash
uv sync
```

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

Most scripts call `load_dotenv()`, so this is picked up automatically.

## Run the agents

```bash
uv run python agents/agent_bot.py
uv run python agents/memory_agent.py
uv run python agents/ReAct_agent.py
uv run python agents/drafter_agent.py
```

## Notes on current behavior

- `agents/memory_agent.py` writes a conversation log to `logging.txt` when you exit.
- `agents/drafter_agent.py` keeps document text in memory, then saves it to a `.txt` file when asked.
- `agents/rag_agent.py` incorporates a vectorDB in the workflow. 

## Dependencies

Key libraries (from `pyproject.toml`):

- `langgraph`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-chroma`
- `langchain-text-splitters`
- `python-dotenv`
