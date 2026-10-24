"""Structured output schemas for ResearchMind AI.

Keep field names stable: rag_graph.py and btw_handler.py depend on them.
Semantic Scholar paper discovery uses its own API metadata and does not
require changing these LangGraph response schemas.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BtwRouteDecision(BaseModel):
    """Decide whether a /btw question requires up-to-date web information."""

    needs_web_search: bool = Field(
        description="True if answering requires current or externally verified information."
    )


class RouterDecision(BaseModel):
    """Select the main ResearchMind AI workflow for a user question."""

    route: Literal["retrieve", "verify_claim", "direct_answer"] = Field(
        description=(
            "retrieve: search research documents or external sources; "
            "verify_claim: check a specific research claim against evidence; "
            "direct_answer: answer a stable general-knowledge question."
        )
    )


class RelevancyDecision(BaseModel):
    """Assess whether retrieved context helps answer the question."""

    is_relevant: bool = Field(
        description="Whether the retrieved passages contain useful information for the question."
    )
    reason: str = Field(
        description="Brief explanation of why the retrieved passages are or are not relevant."
    )


class SupersedingPaper(BaseModel):
    """A research paper identified as challenging or updating a claim."""

    title: str = Field(description="Paper title as it appears in the search evidence.")
    url: str = Field(description="Actual paper URL present in the search evidence.")
    summary: str = Field(
        description="Concise explanation of how the paper challenges, updates, or supersedes the claim."
    )


class ClaimVerificationResult(BaseModel):
    """Structured claim-verification outcome with supporting papers."""

    is_superseded: bool = Field(
        description="True only when the available evidence supports that the claim was superseded."
    )
    verdict_summary: str = Field(
        description=(
            "Concise evidence-based verdict; acknowledge uncertainty or insufficient evidence "
            "rather than treating no search results as proof that a claim remains valid."
        )
    )
    superseding_papers: list[SupersedingPaper] = Field(
        description="Up to three evidence-backed papers, or an empty list if none were established."
    )
