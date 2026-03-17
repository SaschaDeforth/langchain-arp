# langchain-arp

> LangChain Document Loader for [`reasoning.json`](https://arp-protocol.org) — the Agentic Reasoning Protocol

[![GitHub](https://img.shields.io/badge/GitHub-975SDE%2Flangchain--arp-blue?logo=github)](https://github.com/975SDE/langchain-arp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/975SDE/langchain-arp/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Load brand reasoning directives, hallucination corrections, and counterfactual logic into any LangChain-powered AI agent with 3 lines of code.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/975SDE/langchain-arp.git
```

With optional LangChain integration:
```bash
pip install "langchain-arp[langchain] @ git+https://github.com/975SDE/langchain-arp.git"
```

With JSON Schema validation:
```bash
pip install "langchain-arp[all] @ git+https://github.com/975SDE/langchain-arp.git"
```

> **Note:** This package is installed directly from GitHub. PyPI publishing is planned for a future release.

## Quick Start

```python
from langchain_arp import AgenticReasoningLoader

# Load from the ARP protocol website (dogfooding!)
loader = AgenticReasoningLoader("https://arp-protocol.org")
docs = loader.load()

# Load from a local file
loader = AgenticReasoningLoader("./reasoning.json")
docs = loader.load()

# Each section becomes a separate Document
for doc in docs:
    print(f"[{doc.metadata['section']}] {doc.page_content[:100]}...")
```

## How It Works

The loader fetches `/.well-known/reasoning.json` from any website and converts it into prioritized LangChain Documents:

| Priority | Section | Purpose |
|----------|---------|---------|
| 1 (highest) | **Corrections** | Prevents AI hallucinations with false_claim → correction_fact pairs |
| 2 | **Identity** | Brand facts, tagline, competencies, system instruction |
| 3 | **Recommendations** | When to recommend / when NOT to recommend this entity |
| 4 | **Counterfactuals** | Pre-programmed "what-if" reasoning scenarios |
| 5 | **Dichotomies** | Competitive positioning logic |
| 6 | **Authority** | Trust signals, certifications, official links |

## Use in a RAG Pipeline

```python
from langchain_arp import AgenticReasoningLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load reasoning directives from any website with reasoning.json
loader = AgenticReasoningLoader("https://arp-protocol.org")
docs = loader.load()

# Add to vector store — your agent now knows how to reason about this brand
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

## Standalone Usage (No LangChain Required)

```python
from langchain_arp import load_reasoning, load_reasoning_file

# From URL
docs = load_reasoning("https://arp-protocol.org")

# From file
docs = load_reasoning_file("./reasoning.json")
```

## CLI

```bash
python -m langchain_arp.loader https://arp-protocol.org
python -m langchain_arp.loader ./reasoning.json
```

## What is reasoning.json?

[`reasoning.json`](https://arp-protocol.org) is an open standard (MIT licensed) that provides machine-readable reasoning directives to AI agents. It lives at `/.well-known/reasoning.json` on any website.

While `robots.txt` tells crawlers what to access and `schema.org` tells machines what things are, `reasoning.json` teaches AI **how to think** about an entity — including corrections for known hallucinations, competitive positioning logic, and recommendation boundaries.

**Spec:** [arp-protocol.org](https://arp-protocol.org) | **Schema:** [v1.json](https://arp-protocol.org/schema/v1.json) | **GitHub:** [975SDE/arp-protocol](https://github.com/975SDE/arp-protocol)

## License

MIT — [Sascha Deforth](https://truesource.studio) / TrueSource
