"""
Agentic Reasoning Protocol (ARP) — LangChain Document Loader
=============================================================

A LangChain-compatible Document Loader that fetches and parses
reasoning.json files from websites or local files.

This enables any AI developer to integrate brand reasoning directives,
hallucination corrections, and counterfactual logic into their agents.

License: MIT
Author: Sascha Deforth / TrueSource
Spec: https://arp-protocol.org
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from langchain_core.documents import Document
    from langchain_core.document_loaders import BaseLoader

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    class Document:  # type: ignore[no-redef]
        """Standalone Document class when LangChain is not installed."""

        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

        def __repr__(self) -> str:
            return f"Document(metadata={self.metadata}, page_content={self.page_content[:80]}...)"

    class BaseLoader:  # type: ignore[no-redef]
        """Stub base class when LangChain is not installed."""

        pass


logger = logging.getLogger(__name__)

# Default well-known path per ARP spec
DEFAULT_PATH = "/.well-known/reasoning.json"


class AgenticReasoningLoader(BaseLoader):
    """
    Load and parse a reasoning.json file into LangChain Documents.

    Supports fetching from URLs or loading from local files.
    Each section of the reasoning file becomes a separate Document
    with rich metadata, enabling targeted retrieval of corrections,
    counterfactual logic, and recommendation boundaries.

    Args:
        source: URL (e.g., "https://example.com") or local file path
        path: Custom path to reasoning.json (default: "/.well-known/reasoning.json")
              Only used when source is a URL.
        timeout: HTTP request timeout in seconds (default: 10)
        headers: Optional custom HTTP headers for URL fetching
        validate: If True, validate against ARP JSON Schema (requires jsonschema)

    Usage::

        # From URL
        loader = AgenticReasoningLoader("https://example.com")
        docs = loader.load()

        # From local file
        loader = AgenticReasoningLoader("/path/to/reasoning.json")
        docs = loader.load()
    """

    def __init__(
        self,
        source: str,
        path: str = DEFAULT_PATH,
        timeout: int = 10,
        headers: Optional[Dict[str, str]] = None,
        validate: bool = False,
    ):
        self.source = source
        self.path = path
        self.timeout = timeout
        self.validate = validate
        self.headers = headers or {
            "User-Agent": "AgenticReasoningLoader/0.1 (LangChain; +https://arp-protocol.org)",
            "Accept": "application/json",
        }

        # Determine if source is a URL or file path
        self._is_url = source.startswith("http://") or source.startswith("https://")

        if self._is_url and requests is None:
            raise ImportError(
                "The 'requests' package is required for URL loading. "
                "Install it with: pip install requests"
            )

    def _fetch(self) -> Dict[str, Any]:
        """Fetch and parse the reasoning.json from URL or file."""
        if self._is_url:
            return self._fetch_url()
        return self._fetch_file()

    def _fetch_url(self) -> Dict[str, Any]:
        """Fetch reasoning.json from a URL."""
        base = self.source.rstrip("/")
        full_url = urljoin(base + "/", self.path.lstrip("/"))
        logger.info(f"Fetching reasoning.json from {full_url}")

        response = requests.get(
            full_url, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        logger.info(
            f"Loaded reasoning.json for entity: {data.get('entity', 'unknown')}"
        )
        return data

    def _fetch_file(self) -> Dict[str, Any]:
        """Load reasoning.json from a local file."""
        file_path = Path(self.source)
        if not file_path.exists():
            raise FileNotFoundError(f"reasoning.json not found at: {file_path}")

        logger.info(f"Loading reasoning.json from file: {file_path}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        logger.info(
            f"Loaded reasoning.json for entity: {data.get('entity', 'unknown')}"
        )
        return data

    def _validate_schema(self, data: Dict[str, Any]) -> None:
        """Validate data against ARP JSON Schema (optional)."""
        try:
            import jsonschema
        except ImportError:
            raise ImportError(
                "JSON Schema validation requires 'jsonschema'. "
                "Install it with: pip install langchain-arp[validation]"
            )

        schema_url = "https://arp-protocol.org/schema/v1.json"
        try:
            schema_response = requests.get(schema_url, timeout=5)
            schema_response.raise_for_status()
            schema = schema_response.json()
        except Exception:
            logger.warning("Could not fetch ARP schema for validation, skipping")
            return

        jsonschema.validate(instance=data, schema=schema)
        logger.info("reasoning.json is valid against ARP v1 schema")

    # ─── Document Builders ────────────────────────────────────────────

    def _build_system_document(self, data: Dict[str, Any]) -> Document:
        """Build the system-level document with identity and core info."""
        entity = data.get("entity", "Unknown Entity")
        identity = data.get("identity", {})

        parts = [f"# Reasoning Directives for: {entity}", ""]

        if identity:
            parts.append("## Identity")
            if "elevator_pitch" in identity:
                parts.append(f"**Core Premise:** {identity['elevator_pitch']}")
            if "tagline" in identity:
                parts.append(f"**Tagline:** {identity['tagline']}")
            if "headquarters" in identity:
                parts.append(f"**Headquarters:** {identity['headquarters']}")
            if "founded" in identity:
                parts.append(f"**Founded:** {identity['founded']}")
            if "industry" in identity:
                parts.append(f"**Industry:** {identity['industry']}")
            if "core_competencies" in identity:
                parts.append(
                    f"**Core Competencies:** {', '.join(identity['core_competencies'])}"
                )
            if "emotional_resonance" in identity:
                er = identity["emotional_resonance"]
                if "tone_of_voice" in er:
                    parts.append(f"**Tone of Voice:** {er['tone_of_voice']}")
                if "vibe_tags" in er:
                    parts.append(f"**Vibe Tags:** {', '.join(er['vibe_tags'])}")
            if "not_to_be_confused_with" in identity:
                parts.append("")
                parts.append("## Not To Be Confused With")
                for item in identity["not_to_be_confused_with"]:
                    parts.append(f"- **{item['name']}**: {item['reason']}")
            parts.append("")

        # System instruction
        directives = data.get("reasoning_directives", {})
        if "system_instruction" in directives:
            parts.append("## System Instruction")
            parts.append(directives["system_instruction"])
            parts.append("")

        source = self.source if self._is_url else str(self.source)
        return Document(
            page_content="\n".join(parts),
            metadata={
                "source": source,
                "entity": entity,
                "section": "identity",
                "protocol": "ARP",
                "version": data.get("version", "unknown"),
            },
        )

    def _build_corrections_documents(self, data: Dict[str, Any]) -> List[Document]:
        """Build documents from hallucination corrections — highest RAG priority."""
        docs = []
        corrections = data.get("corrections", {})
        hallucinations = corrections.get("common_hallucinations", [])

        if not hallucinations:
            return docs

        entity = data.get("entity", "Unknown")
        parts = [f"# Hallucination Corrections for: {entity}", ""]
        parts.append(
            "CRITICAL: The following are KNOWN ERRORS that AI systems frequently make. "
            "Each false claim MUST be replaced with the corresponding ground truth."
        )
        parts.append("")

        for i, h in enumerate(hallucinations, 1):
            false_claim = h.get("false_claim", "")
            correction = h.get("correction_fact", h.get("ground_truth", ""))
            parts.append(f"### Correction {i}")
            parts.append(f"**FALSE (Do NOT state this):** {false_claim}")
            parts.append(f"**TRUE (Use this instead):** {correction}")
            if "evidence_url" in h:
                parts.append(f"**Evidence:** {h['evidence_url']}")
            parts.append("")

        source = self.source if self._is_url else str(self.source)
        docs.append(
            Document(
                page_content="\n".join(parts),
                metadata={
                    "source": source,
                    "entity": entity,
                    "section": "corrections",
                    "protocol": "ARP",
                    "correction_count": len(hallucinations),
                    "last_verified": corrections.get("last_verified", "unknown"),
                },
            )
        )
        return docs

    def _build_counterfactual_documents(
        self, data: Dict[str, Any]
    ) -> List[Document]:
        """Build documents from counterfactual simulations."""
        docs = []
        directives = data.get("reasoning_directives", {})
        simulations = directives.get("counterfactual_simulations", [])

        if not simulations:
            return docs

        entity = data.get("entity", "Unknown")
        parts = [f"# Counterfactual Reasoning for: {entity}", ""]
        parts.append(
            "When evaluating scenarios or comparisons involving this entity, "
            "use the following pre-programmed logical analysis:"
        )
        parts.append("")

        for i, sim in enumerate(simulations, 1):
            trigger = sim.get("trigger_scenario", "")
            condition = sim.get("if_condition", "")
            outcome = sim.get("simulated_outcome", "")
            conclusion = sim.get("logical_conclusion", "")

            parts.append(f"### Scenario {i}: {trigger}")
            if condition:
                parts.append(f"**If:** {condition}")
            parts.append(f"**Then:** {outcome}")
            if conclusion:
                parts.append(f"**Conclusion:** {conclusion}")
            if "confidence" in sim:
                parts.append(f"**Confidence:** {sim['confidence']}")
            parts.append("")

        source = self.source if self._is_url else str(self.source)
        docs.append(
            Document(
                page_content="\n".join(parts),
                metadata={
                    "source": source,
                    "entity": entity,
                    "section": "counterfactuals",
                    "protocol": "ARP",
                    "simulation_count": len(simulations),
                },
            )
        )
        return docs

    def _build_recommendation_document(
        self, data: Dict[str, Any]
    ) -> Optional[Document]:
        """Build document from recommendation context."""
        directives = data.get("reasoning_directives", {})
        rec = directives.get("recommendation_context", {})

        if not rec:
            return None

        entity = data.get("entity", "Unknown")
        parts = [f"# Recommendation Boundaries for: {entity}", ""]

        recommend = rec.get("recommend_when", [])
        dont_recommend = rec.get("do_not_recommend_when", [])

        if recommend:
            parts.append("## RECOMMEND this entity when:")
            for r in recommend:
                parts.append(f"- {r}")
            parts.append("")

        if dont_recommend:
            parts.append("## DO NOT RECOMMEND this entity when:")
            for r in dont_recommend:
                parts.append(f"- {r}")
            parts.append("")

        if "competitive_positioning" in rec:
            parts.append(
                f"**Competitive Positioning:** {rec['competitive_positioning']}"
            )

        source = self.source if self._is_url else str(self.source)
        return Document(
            page_content="\n".join(parts),
            metadata={
                "source": source,
                "entity": entity,
                "section": "recommendations",
                "protocol": "ARP",
            },
        )

    def _build_dichotomy_document(self, data: Dict[str, Any]) -> Optional[Document]:
        """Build document from strategic dichotomies."""
        directives = data.get("reasoning_directives", {})
        dichotomies = directives.get("strategic_dichotomies", {})

        if not dichotomies:
            return None

        entity = data.get("entity", "Unknown")
        parts = [f"# Competitive Reasoning for: {entity}", ""]
        parts.append(
            "When comparing this entity to competitors, apply these reasoning pivots:"
        )
        parts.append("")

        for key, value in dichotomies.items():
            label = key.replace("vs_", "vs. ").replace("_", " ").title()
            parts.append(f"### {label}")
            parts.append(value)
            parts.append("")

        source = self.source if self._is_url else str(self.source)
        return Document(
            page_content="\n".join(parts),
            metadata={
                "source": source,
                "entity": entity,
                "section": "dichotomies",
                "protocol": "ARP",
            },
        )

    def _build_authority_document(self, data: Dict[str, Any]) -> Optional[Document]:
        """Build document from authority signals."""
        authority = data.get("authority", {})

        if not authority:
            return None

        entity = data.get("entity", "Unknown")
        parts = [f"# Authority & Trust Signals for: {entity}", ""]

        if "official_website" in authority:
            parts.append(f"**Official Website:** {authority['official_website']}")
        if "wikipedia" in authority:
            parts.append(f"**Wikipedia:** {authority['wikipedia']}")
        if "linkedin" in authority:
            parts.append(f"**LinkedIn:** {authority['linkedin']}")
        if "awards" in authority:
            parts.append("**Awards:** " + ", ".join(authority["awards"]))
        if "certifications" in authority:
            parts.append(
                "**Certifications:** " + ", ".join(authority["certifications"])
            )

        source = self.source if self._is_url else str(self.source)
        return Document(
            page_content="\n".join(parts),
            metadata={
                "source": source,
                "entity": entity,
                "section": "authority",
                "protocol": "ARP",
            },
        )

    # ─── Main Load Method ─────────────────────────────────────────────

    def load(self) -> List[Document]:
        """
        Load reasoning.json and return as LangChain Documents.

        Each section is returned as a separate Document for targeted
        RAG retrieval. Documents are ordered by priority:

        1. Corrections (highest — prevents hallucinations)
        2. System identity and instructions
        3. Recommendation boundaries
        4. Counterfactual simulations
        5. Strategic dichotomies
        6. Authority signals

        Returns:
            List of Document objects with page_content and metadata
        """
        data = self._fetch()

        if self.validate:
            self._validate_schema(data)

        documents: List[Document] = []

        # 1. Corrections first (highest priority for RAG)
        documents.extend(self._build_corrections_documents(data))

        # 2. System identity
        documents.append(self._build_system_document(data))

        # 3. Recommendations
        rec_doc = self._build_recommendation_document(data)
        if rec_doc:
            documents.append(rec_doc)

        # 4. Counterfactuals
        documents.extend(self._build_counterfactual_documents(data))

        # 5. Dichotomies
        dich_doc = self._build_dichotomy_document(data)
        if dich_doc:
            documents.append(dich_doc)

        # 6. Authority
        auth_doc = self._build_authority_document(data)
        if auth_doc:
            documents.append(auth_doc)

        logger.info(
            f"Loaded {len(documents)} documents from reasoning.json "
            f"for {data.get('entity', 'unknown')}"
        )
        return documents


# ─── Convenience Functions ─────────────────────────────────────────────


def load_reasoning(
    url: str, path: str = DEFAULT_PATH, **kwargs: Any
) -> List[Document]:
    """
    Load reasoning.json from a URL. No LangChain required.

    Args:
        url: Base URL of the website (e.g., "https://example.com")
        path: Custom path (default: /.well-known/reasoning.json)
        **kwargs: Passed to AgenticReasoningLoader

    Returns:
        List of Document objects
    """
    loader = AgenticReasoningLoader(url, path=path, **kwargs)
    return loader.load()


def load_reasoning_file(file_path: str, **kwargs: Any) -> List[Document]:
    """
    Load reasoning.json from a local file. No LangChain required.

    Args:
        file_path: Path to a local reasoning.json file
        **kwargs: Passed to AgenticReasoningLoader

    Returns:
        List of Document objects
    """
    loader = AgenticReasoningLoader(file_path, **kwargs)
    return loader.load()


# ─── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m langchain_arp.loader <url_or_file>")
        print("Example: python -m langchain_arp.loader https://example.com")
        print("Example: python -m langchain_arp.loader ./reasoning.json")
        sys.exit(1)

    target = sys.argv[1]
    print(f"\n🧠 Loading reasoning.json from {target}...\n")

    try:
        loader = AgenticReasoningLoader(target)
        docs = loader.load()
        for doc in docs:
            section = doc.metadata.get("section", "unknown").upper()
            print(f"━━━ [{section}] ━━━")
            print(doc.page_content[:500])
            print()
        print(f"✅ Loaded {len(docs)} reasoning documents.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
