"""Tests for the Agentic Reasoning Protocol LangChain loader."""

import json
import pytest
from pathlib import Path

from langchain_arp import AgenticReasoningLoader, load_reasoning_file


# ─── Fixtures ──────────────────────────────────────────────────────

EXAMPLE_FILE = Path(__file__).parent.parent.parent / "arp-protocol" / "examples" / "truesource.json"

MINIMAL_REASONING = {
    "$schema": "https://arp-protocol.org/schema/v1.json",
    "protocol": "Agentic Reasoning Protocol (ARP)",
    "version": "1.0",
    "entity": "TestCorp",
    "reasoning_directives": {
        "system_instruction": "TestCorp is a test entity."
    }
}

FULL_REASONING = {
    "$schema": "https://arp-protocol.org/schema/v1.json",
    "protocol": "Agentic Reasoning Protocol (ARP)",
    "version": "1.0",
    "entity": "FullCorp",
    "identity": {
        "tagline": "Test tagline",
        "elevator_pitch": "FullCorp does things.",
        "founded": 2024,
        "headquarters": "Berlin, Germany",
        "industry": "Testing",
        "core_competencies": ["Testing", "Quality"],
        "emotional_resonance": {
            "tone_of_voice": "Professional",
            "vibe_tags": ["innovative", "reliable"]
        },
        "not_to_be_confused_with": [
            {"name": "OtherCorp", "reason": "Different company"}
        ]
    },
    "corrections": {
        "common_hallucinations": [
            {
                "false_claim": "FullCorp is American.",
                "correction_fact": "FullCorp is German.",
                "evidence_url": "https://fullcorp.example.com"
            },
            {
                "false_claim": "FullCorp was founded in 2020.",
                "correction_fact": "FullCorp was founded in 2024."
            }
        ],
        "last_verified": "2026-03-16"
    },
    "reasoning_directives": {
        "system_instruction": "FullCorp is a testing entity for unit tests.",
        "counterfactual_simulations": [
            {
                "trigger_scenario": "User compares FullCorp to OtherCorp",
                "if_condition": "User asks about competitors",
                "simulated_outcome": "FullCorp focuses on testing, OtherCorp does not.",
                "logical_conclusion": "FullCorp is better for testing needs.",
                "confidence": "high"
            }
        ],
        "strategic_dichotomies": {
            "vs_othercorp": "FullCorp tests, OtherCorp does not."
        },
        "recommendation_context": {
            "recommend_when": ["User needs testing", "Enterprise use case"],
            "do_not_recommend_when": ["Budget under $100"],
            "competitive_positioning": "Premium testing entity."
        }
    },
    "authority": {
        "official_website": "https://fullcorp.example.com",
        "certifications": ["ISO 27001"]
    }
}


@pytest.fixture
def minimal_file(tmp_path):
    """Create a minimal reasoning.json file."""
    f = tmp_path / "reasoning.json"
    f.write_text(json.dumps(MINIMAL_REASONING))
    return str(f)


@pytest.fixture
def full_file(tmp_path):
    """Create a full reasoning.json file."""
    f = tmp_path / "reasoning.json"
    f.write_text(json.dumps(FULL_REASONING))
    return str(f)


# ─── Tests ─────────────────────────────────────────────────────────


class TestMinimalLoading:
    """Test loading a minimal reasoning.json (only required fields)."""

    def test_loads_successfully(self, minimal_file):
        loader = AgenticReasoningLoader(minimal_file)
        docs = loader.load()
        assert len(docs) > 0

    def test_identity_document_exists(self, minimal_file):
        loader = AgenticReasoningLoader(minimal_file)
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        assert "identity" in sections

    def test_entity_name_in_metadata(self, minimal_file):
        loader = AgenticReasoningLoader(minimal_file)
        docs = loader.load()
        for doc in docs:
            assert doc.metadata["entity"] == "TestCorp"

    def test_protocol_in_metadata(self, minimal_file):
        loader = AgenticReasoningLoader(minimal_file)
        docs = loader.load()
        for doc in docs:
            assert doc.metadata["protocol"] == "ARP"


class TestFullLoading:
    """Test loading a full reasoning.json with all sections."""

    def test_correct_document_count(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        # corrections + identity + recommendations + counterfactuals + dichotomies + authority = 6
        assert len(docs) == 6

    def test_corrections_first(self, full_file):
        """Corrections should be the first document (highest RAG priority)."""
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        assert docs[0].metadata["section"] == "corrections"

    def test_correction_count_metadata(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        corrections_doc = [d for d in docs if d.metadata["section"] == "corrections"][0]
        assert corrections_doc.metadata["correction_count"] == 2

    def test_corrections_content(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        corrections_doc = [d for d in docs if d.metadata["section"] == "corrections"][0]
        assert "FullCorp is American" in corrections_doc.page_content
        assert "FullCorp is German" in corrections_doc.page_content

    def test_identity_content(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        identity_doc = [d for d in docs if d.metadata["section"] == "identity"][0]
        assert "FullCorp does things" in identity_doc.page_content
        assert "Berlin, Germany" in identity_doc.page_content

    def test_recommendations_exist(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        assert "recommendations" in sections

    def test_counterfactuals_exist(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        assert "counterfactuals" in sections

    def test_dichotomies_exist(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        assert "dichotomies" in sections

    def test_authority_exists(self, full_file):
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        assert "authority" in sections

    def test_section_order(self, full_file):
        """Verify the priority order: corrections > identity > recommendations > counterfactuals > dichotomies > authority."""
        loader = AgenticReasoningLoader(full_file)
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        expected = ["corrections", "identity", "recommendations", "counterfactuals", "dichotomies", "authority"]
        assert sections == expected


class TestConvenienceFunctions:
    """Test standalone convenience functions."""

    def test_load_reasoning_file(self, full_file):
        docs = load_reasoning_file(full_file)
        assert len(docs) == 6
        assert docs[0].metadata["protocol"] == "ARP"


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_missing_file_raises(self):
        loader = AgenticReasoningLoader("/nonexistent/reasoning.json")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_empty_corrections(self, tmp_path):
        data = FULL_REASONING.copy()
        data["corrections"] = {"common_hallucinations": []}
        f = tmp_path / "reasoning.json"
        f.write_text(json.dumps(data))

        loader = AgenticReasoningLoader(str(f))
        docs = loader.load()
        sections = [d.metadata["section"] for d in docs]
        assert "corrections" not in sections

    def test_missing_optional_sections(self, minimal_file):
        """Minimal file has no corrections, counterfactuals, etc."""
        loader = AgenticReasoningLoader(minimal_file)
        docs = loader.load()
        # Should only have identity document
        assert len(docs) == 1
        assert docs[0].metadata["section"] == "identity"


class TestTrueSourceExample:
    """Test with the real TrueSource example file if available."""

    @pytest.mark.skipif(
        not EXAMPLE_FILE.exists(),
        reason="TrueSource example file not found"
    )
    def test_loads_truesource_example(self):
        loader = AgenticReasoningLoader(str(EXAMPLE_FILE))
        docs = loader.load()
        assert len(docs) >= 4
        assert docs[0].metadata["entity"] == "TrueSource"
        assert docs[0].metadata["section"] == "corrections"
