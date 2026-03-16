"""
Basic usage example for langchain-arp.

This example loads a reasoning.json file and prints
all reasoning documents with their sections.
"""

from langchain_arp import AgenticReasoningLoader

# ─── Option 1: Load from a URL ────────────────────────────────────
# loader = AgenticReasoningLoader("https://example.com")

# ─── Option 2: Load from a local file ─────────────────────────────
loader = AgenticReasoningLoader("../arp-protocol/examples/truesource.json")

# Load all reasoning documents
docs = loader.load()

print(f"🧠 Loaded {len(docs)} reasoning documents\n")

for doc in docs:
    section = doc.metadata.get("section", "unknown").upper()
    entity = doc.metadata.get("entity", "unknown")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📋 Section: {section}")
    print(f"🏢 Entity:  {entity}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(doc.page_content[:300])
    print(f"\n... ({len(doc.page_content)} chars total)")
    print()

# ─── Use in a RAG pipeline ────────────────────────────────────────
#
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
#
# vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
# retriever = vectorstore.as_retriever()
#
# # Now your AI agent knows how to reason about this brand!
