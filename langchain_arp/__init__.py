"""
langchain-arp — LangChain Document Loader for reasoning.json
============================================================

Load reasoning.json files (Agentic Reasoning Protocol) into LangChain Documents
for RAG pipelines, AI agents, and brand reasoning applications.

Usage:
    from langchain_arp import AgenticReasoningLoader

    loader = AgenticReasoningLoader("https://example.com")
    docs = loader.load()

Standalone (no LangChain required):
    from langchain_arp import load_reasoning

    docs = load_reasoning("https://example.com")
"""

from langchain_arp.loader import AgenticReasoningLoader, load_reasoning, load_reasoning_file

__version__ = "0.1.0"
__all__ = ["AgenticReasoningLoader", "load_reasoning", "load_reasoning_file"]
