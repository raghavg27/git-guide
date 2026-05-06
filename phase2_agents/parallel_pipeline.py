"""
phase2_agents/parallel_pipeline.py
────────────────────────────────────
Async parallel pre-processing: intent classification + query rewriting.

WHY:
  Intent classification and query rewriting are independent — both only
  need the raw user query. Running them sequentially in CrewAI wastes 3-5s
  waiting for the first call to finish before the second can start.

  asyncio.gather() fires both simultaneously. Wall-clock time becomes
  max(intent_time, rewrite_time) instead of their sum.

BEFORE (sequential CrewAI):  intent(3-5s) → rewrite(3-5s) = 6-10s
AFTER  (parallel async):     intent(3-5s)
                              rewrite(3-5s)  ← concurrent
                              = ~3-5s total

USAGE:
  import asyncio
  from phase2_agents.parallel_pipeline import run_parallel_preprocess

  context = asyncio.run(run_parallel_preprocess(query))
  enriched_query = f"{query}\n\n[PREPROCESSING CONTEXT]\n{context}"
  crew.kickoff(inputs={"user_query": enriched_query})
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from config.llm_client import async_get_llm_response


_INTENT_SYSTEM_PROMPT = """\
You are an Intent Classifier for a GitLab documentation chatbot.
Classify the user's question into one of:
  DEBUGGING, HOW_TO, EXPLANATION, TROUBLESHOOT, API, MIGRATION, PERMISSIONS, VERSION_INFO, UNKNOWN

Output:
  Intent: <CATEGORY>
  Confidence: <HIGH|MEDIUM|LOW>
  Key Concepts: <comma-separated technical terms>
  Likely Section: <ci|api|admin|runner|user|security|topics|update>
  Has Code: <YES|NO>
  Analysis: <one sentence reasoning>\
"""

_REWRITE_SYSTEM_PROMPT = """\
You are a Query Rewriter for a GitLab documentation search system.
Rewrite the user's question into precise, search-optimized versions.

Output:
  Primary Search Query: <optimised query>
  Alternative Phrasings:
    1. <alt 1>
    2. <alt 2>
    3. <alt 3>
  Filter Recommendations:
    Section: <ci|api|admin|runner|user|security|topics|update|none>
    Code Examples: <YES|NO|OPTIONAL>
  Confidence: <HIGH|MEDIUM|LOW>
  Rationale: <one sentence reasoning>\
"""


async def _classify_intent(query: str) -> str:
    return await async_get_llm_response(
        prompt=f"Classify the intent of this question: {query}",
        system_prompt=_INTENT_SYSTEM_PROMPT,
        max_tokens=300,
    )


async def _rewrite_query(query: str) -> str:
    return await async_get_llm_response(
        prompt=f"Rewrite this question for documentation search: {query}",
        system_prompt=_REWRITE_SYSTEM_PROMPT,
        max_tokens=400,
    )


async def run_parallel_preprocess(query: str) -> str:
    """
    Fire intent classification and query rewriting simultaneously.

    Returns a combined context string to inject into the retriever's task,
    giving it the same enriched context the old sequential agents produced —
    but in roughly half the time.
    """
    intent_result, rewrite_result = await asyncio.gather(
        _classify_intent(query),
        _rewrite_query(query),
    )
    return (
        f"INTENT ANALYSIS:\n{intent_result}\n\n"
        f"QUERY REWRITES:\n{rewrite_result}"
    )
