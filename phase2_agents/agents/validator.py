"""
phase2_agents/agents/validator.py
──────────────────────────────────
Heuristic chunk validator — no LLM, runs in ~0ms.

The old CrewAI Validator agent made a 5-8s LLM call to assess chunk quality.
All the information it needed (relevance score, is_deprecated, text length)
is already present in the RetrievalResult metadata, so an LLM is unnecessary.

Validation is now applied directly inside _format_retrieval_results() in
retrieval_tools.py using these rules:

  relevance_score < 0.40  → SKIP  (dropped, never shown to synthesiser)
  is_deprecated == True   → WARN  (shown with deprecation caveat)
  len(text) < 80 chars    → WARN  (stub/heading, may lack context)
  otherwise               → KEEP

The validate_chunks() function below is exported for standalone use or testing.
The CrewAI agent and crew task have been removed — see crew.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1_ingestion.vector_store import RetrievalResult
from phase2_agents.tools.retrieval_tools import RELEVANCE_THRESHOLD, MIN_TEXT_LENGTH


def validate_chunks(
    results: list[RetrievalResult],
) -> tuple[list[RetrievalResult], list[tuple[RetrievalResult, str]], list[RetrievalResult]]:
    """
    Apply heuristic validation to a list of RetrievalResult objects.

    Returns:
        keep  — clean chunks, ready for the synthesiser
        warn  — usable chunks with a caveat (deprecated, short)
        skip  — irrelevant chunks dropped from the answer
    """
    keep: list[RetrievalResult] = []
    warn: list[tuple[RetrievalResult, str]] = []
    skip: list[RetrievalResult] = []

    for r in results:
        if r.relevance_score < RELEVANCE_THRESHOLD:
            skip.append(r)
        elif r.is_deprecated:
            warn.append((r, "marked deprecated in GitLab docs"))
        elif len(r.text.strip()) < MIN_TEXT_LENGTH:
            warn.append((r, "chunk is very short — may lack context"))
        else:
            keep.append(r)

    return keep, warn, skip
