"""
phase2_agents/agents/retriever.py
──────────────────────────────────
Agent 3: Retriever

WHAT IT DOES:
  Executes the search(es) against the vector store.
  Uses the Query Rewriter's recommendations to decide strategy.

SEARCH STRATEGIES:
  1. SIMPLE SEMANTIC SEARCH
     One query → top 5 results
     Use when: clear, focused question
  
  2. FILTERED SEARCH
     Query + metadata filters (section, has_code, etc.)
     Use when: question specifies a topic area
  
  3. MULTI-QUERY FUSION
     3+ query variations + fuse results with RRF
     Use when: complex or multi-faceted question

TOOLS AVAILABLE:
  - semantic_search()      — broad semantic search
  - filtered_search()      — search with metadata constraints
  - multi_query_search()   — generate + fuse multiple queries
  - get_chunk_by_id()      — fetch specific chunk by ID

DECISION LOGIC:
  if query_rewriter_suggested_filters:
    use filtered_search()
  elif is_complex_question:
    use multi_query_search()
  else:
    use semantic_search()

OUTPUT:
  "Retrieved 5 chunks:
   
   [1] GitLab CI Caching - 98% relevance
       → https://docs.gitlab.com/ee/ci/caching/
   
   [2] Docker Layer Caching - 87% relevance
       → https://docs.gitlab.com/ee/ci/docker/...
   
   [3] ..."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crewai import Agent
from config.llm_client import get_llm_client
from phase2_agents.tools.retrieval_tools import RETRIEVAL_TOOLS


def create_retriever_agent() -> Agent:
    """Create the Retriever agent with access to search tools."""
    
    system_prompt = """You are a Retrieval Specialist for a GitLab documentation system.

Your job is to search the documentation and retrieve the most relevant chunks.

YOU HAVE ACCESS TO MULTIPLE SEARCH STRATEGIES:
  1. semantic_search()                          ← DEFAULT, use this first
     - Cost: ~0.15s (local embedding only, no LLM)
     - Use for: any clear, well-formed question (5+ words with a clear subject)
     - Returns: top 5 relevant chunks by semantic similarity

  2. filtered_search(section="...", has_code=True)
     - Cost: ~0.15s (local embedding + metadata filter)
     - Use when: the question is clearly about one section (ci, api, runner, etc.)
     - Available sections: ci, api, admin, runner, user, security, topics, update

  3. multi_query_search()                       ← LAST RESORT ONLY
     - Cost: 3-5s extra (fires an LLM call to generate phrasings)
     - Use ONLY when: query is very short (< 5 words) or clearly vague/ambiguous
     - Examples where it helps: "caching", "runner broken", "pipeline slow"
     - DO NOT use for clear questions — the tool itself detects specific queries
       and skips the LLM call, making it identical to semantic_search anyway

  4. get_chunk_by_id(chunk_id)
     - Use when: you need to fetch a specific chunk by ID

DECISION LOGIC:
  DEFAULT → semantic_search()   (fast, handles most questions well)
  if question clearly targets one section (ci, api, runner, etc.):
    → filtered_search(section=...)
  if query is very short (< 5 words) or has no clear subject:
    → multi_query_search()   (only case where the LLM phrasing cost is justified)

WHEN TO RETRY:
  if first search returns no results → broaden the query, try semantic_search again
  if results seem off-topic → try filtered_search with a specific section
  avoid multi_query_search as a retry for clear questions — it adds 3-5s and
  the tool will skip its LLM call anyway, making it identical to semantic_search"""
    
    agent = Agent(
        role="Retrieval Specialist",
        goal=(
            "Execute precise, effective searches to retrieve the most relevant "
            "documentation chunks for user questions. Use multiple search strategies "
            "to ensure good coverage."
        ),
        backstory=(
            "You are an expert at search and information retrieval. "
            "You know how to use different search strategies for different questions. "
            "You understand that semantic search, filtering, and query fusion each "
            "have their strengths. You iterate and refine searches to get the best results."
        ),
        llm=get_llm_client(),
        tools=RETRIEVAL_TOOLS,
        verbose=True,
        allow_delegation=False,
    )
    
    return agent
