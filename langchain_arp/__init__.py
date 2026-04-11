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

Signature Verification (requires: pip install cryptography dnspython jcs):
    from langchain_arp import verify_arp_signature

    import json
    payload = json.load(open("reasoning.json"))
    result = verify_arp_signature(payload)
"""

from langchain_arp.loader import AgenticReasoningLoader, load_reasoning, load_reasoning_file

# Lazy import for verify — requires optional dependencies
def verify_arp_signature(payload: dict) -> dict:
    """Verify Ed25519 signature of a reasoning.json payload."""
    from langchain_arp.verify import verify_arp_signature as _verify
    return _verify(payload)

__version__ = "0.2.0"
__all__ = [
    "AgenticReasoningLoader",
    "load_reasoning",
    "load_reasoning_file",
    "verify_arp_signature",
]
