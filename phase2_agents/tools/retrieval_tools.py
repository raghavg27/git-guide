"""
phase2_agents/tools/retrieval_tools.py
──────────────────────────────────────
Tools that agents use to search the vector store.

These are wrapped as CrewAI tools — agents call them as functions.
Each tool handles one specific retrieval scenario.

WHY MULTIPLE TOOLS?
  Different queries need different retrieval strategies:
  - Semantic search (default): "how do I cache npm packages?"
  - Filtered search: "show me CI/CD docs only"
  - Multi-query fusion: combine 3 phrasings for better recall
  - ID lookup: retrieve a specific chunk by ID (for citations)

HOW AGENTS USE THESE:
  @tool decorator makes them available to agents.
  Agent calls: tools.semantic_search("my query")
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crewai.tools import tool
from config.settings import settings
from config.llm_client import get_llm_response
from phase1_ingestion.embedder import DocEmbedder
from phase1_ingestion.vector_store import VectorStore, RetrievalResult

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCES (loaded once, reused across all tools)
# ─────────────────────────────────────────────────────────────────────────────

_embedder: Optional[DocEmbedder] = None
_vector_store: Optional[VectorStore] = None


def _get_embedder() -> DocEmbedder:
    """Lazy-load embedder (first call downloads model if needed)."""
    global _embedder
    if _embedder is None:
        _embedder = DocEmbedder()
    return _embedder


def _get_vector_store() -> VectorStore:
    """Lazy-load vector store (first call connects to ChromaDB)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.connect()
    return _vector_store


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool("semantic_search")
def semantic_search(query: str, n_results: int = 5) -> str:
    """
    Search GitLab documentation using semantic similarity.
    
    This is the PRIMARY retrieval method. It understands meaning:
    - "how to fix pipeline timeout" matches "GitLab CI timeout troubleshooting"
    - Ignores exact keyword matching
    
    Args:
        query: The question or topic to search for
        n_results: Number of results to return (default 5, max 15)
    
    Returns:
        Formatted search results with sources and relevance scores
    
    Example:
        >>> semantic_search("how do I cache node_modules in CI?")
        "📄 Found 5 relevant chunks:
         
         [1] Page: CI/CD caching (score: 95%)
         Section: Cache configuration
         URL: https://docs.gitlab.com/ee/ci/caching/
         Text: ...
         
         [2] ..."
    """
    if not query or len(query.strip()) < 3:
        return "❌ Query too short. Please ask a complete question."
    
    # Cap n_results to prevent abuse
    n_results = min(int(n_results), 15)
    
    embedder = _get_embedder()
    store = _get_vector_store()
    
    # Embed the query
    query_embedding = embedder.embed_query(query)
    
    # Search without filters (broad search)
    results = store.query(
        query_embedding,
        n_results=n_results,
        exclude_deprecated=True,  # Always skip deprecated content
    )
    
    if not results:
        return (
            f"❌ No relevant documentation found for: {query!r}\n"
            f"Try rephrasing your question or search for a broader topic."
        )
    
    return _format_retrieval_results(results, query)


@tool("filtered_search")
def filtered_search(
    query: str,
    section: Optional[str] = None,
    has_code: bool = False,
    n_results: int = 5,
) -> str:
    """
    Search with metadata filters applied.
    
    Use this when you know the context:
    - Only search CI/CD docs: section="ci"
    - Only chunks with code examples: has_code=True
    - Combine filters for precision
    
    Available sections:
      - ci          (CI/CD pipelines, YAML, runners)
      - api         (REST & GraphQL API)
      - user        (merge requests, branches, permissions)
      - runner      (GitLab Runner configuration)
      - admin       (administration, self-managed)
      - security    (security scanning, SAST, DAST)
      - topics      (general topics)
      - update      (upgrade guides, version info)
    
    Args:
        query: Search query
        section: Limit to one section (optional)
        has_code: Only return chunks with code examples
        n_results: Number of results
    
    Returns:
        Formatted results (or message if none found)
    """
    embedder = _get_embedder()
    store = _get_vector_store()
    
    query_embedding = embedder.embed_query(query)
    n_results = min(int(n_results), 15)
    
    results = store.filtered_query(
        query_embedding,
        section=section,
        has_code=has_code,
        exclude_deprecated=True,
        n_results=n_results,
    )
    
    if not results:
        filters_desc = []
        if section:
            filters_desc.append(f"section={section}")
        if has_code:
            filters_desc.append("has_code=True")
        
        return (
            f"❌ No results found for: {query!r}\n"
            f"Applied filters: {', '.join(filters_desc) or 'none'}\n"
            f"Try broadening your search or removing filters."
        )
    
    return _format_retrieval_results(results, query)


_CLEAR_QUESTION_STARTERS = {
    "how", "what", "why", "when", "where", "which", "can", "does",
    "is", "are", "will", "should", "would",
}


def _is_ambiguous_query(query: str) -> bool:
    """
    True  → short/vague query; worth paying the LLM cost to expand phrasings.
    False → well-formed query; a single embedding retrieves it just fine.

    Rules:
      < 5 words                            → True  ("caching", "runner broken")
      starts with question word + 5+ words → False ("how do I cache npm packages?")
      7+ words                             → False (long queries are specific)
      everything else                      → True  (medium-length, unclear structure)
    """
    words = query.lower().split()
    n = len(words)

    if n < 5:
        return True
    if words[0] in _CLEAR_QUESTION_STARTERS and n >= 5:
        return False
    if n >= 7:
        return False
    return True


@tool("multi_query_search")
def multi_query_search(query: str, n_results: int = 5) -> str:
    """
    Generate multiple phrasings of the query and fuse results.

    WHEN TO USE: Only for short (< 5 words) or clearly vague queries where
    the original phrasing alone is unlikely to retrieve the right docs.
    Examples: "caching", "runner broken", "pipeline slow"

    For clear, well-formed questions (5+ words with a clear subject) this tool
    automatically skips the LLM phrasing call and runs a direct semantic search
    instead — same quality, 3-5s faster.

    Args:
        query: Original user question
        n_results: Results to return
    """
    import json

    n_results = min(int(n_results), 15)
    embedder = _get_embedder()
    store = _get_vector_store()

    # Guard: skip the LLM phrasing call for well-formed queries.
    # A specific question retrieves well with a single embedding (0.15s).
    # The LLM phrasing call costs 3-5s and only helps for vague/short input.
    if not _is_ambiguous_query(query):
        query_embedding = embedder.embed_query(query)
        results = store.query(query_embedding, n_results=n_results, exclude_deprecated=True)
        if not results:
            return f"❌ No documentation found for: {query!r}"
        note = "ℹ️  Query is specific — used direct semantic search (LLM phrasing skipped)\n\n"
        return note + _format_retrieval_results(results, query)

    # Ambiguous query — generate alternative phrasings via LLM
    system_prompt = (
        "You are a query rewriter for a documentation search system. "
        "Given a user question, generate 2-3 alternative phrasings "
        "that would retrieve the same information. "
        'Format your response as a JSON list: ["phrasing1", "phrasing2", "phrasing3"]'
    )
    try:
        response = get_llm_response(
            prompt=f"Generate alternative phrasings for: {query}",
            system_prompt=system_prompt,
            max_tokens=150,
        )
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        if start_idx >= 0 and end_idx > start_idx:
            phrasings = json.loads(response[start_idx:end_idx])
        else:
            phrasings = [query]
    except Exception:
        phrasings = [query]

    if query not in phrasings:
        phrasings = [query] + phrasings[:2]
    else:
        phrasings = phrasings[:3]

    query_embeddings = embedder.embed_queries(phrasings)
    results = store.multi_query(
        query_embeddings,
        n_results_per_query=n_results // len(phrasings) + 1,
        exclude_deprecated=True,
    )
    results = results[:n_results]

    if not results:
        return f"❌ Multi-query search found no results for: {query!r}"

    header = f"📚 Multi-query results ({len(phrasings)} phrasings fused):\n\n"
    return header + _format_retrieval_results(results, query)


@tool("get_chunk_by_id")
def get_chunk_by_id(chunk_id: str) -> str:
    """
    Retrieve a specific chunk by its ID.
    
    Used by the Validator and Synthesiser agents to fetch full text
    of a chunk they've already identified.
    
    Args:
        chunk_id: The chunk identifier (e.g., "ci__yaml__index__5")
    
    Returns:
        Full chunk text with metadata
    """
    store = _get_vector_store()
    result = store.get_by_id(chunk_id)
    
    if not result:
        return f"❌ Chunk not found: {chunk_id}"
    
    return (
        f"📄 {result.page_title}\n"
        f"Section: {result.chunk_header}\n"
        f"URL: {result.doc_url}\n"
        f"Version: {result.gitlab_version_introduced or 'n/a'}\n"
        f"Deprecated: {'Yes ⚠️' if result.is_deprecated else 'No'}\n\n"
        f"{result.text}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC VALIDATION (replaces the LLM-based Validator agent)
# ─────────────────────────────────────────────────────────────────────────────

# Chunks below this cosine similarity score are dropped before the synthesiser
# sees them. Saves the 5-8s LLM call that the old Validator agent made.
RELEVANCE_THRESHOLD = 0.40

# Chunks shorter than this are flagged as stubs (likely a heading or redirect)
MIN_TEXT_LENGTH = 80


def _validate_chunk(result: RetrievalResult) -> tuple[str, str]:
    """
    Apply heuristic rules to a single chunk.

    Returns (status, caveat) where status is KEEP / WARN / SKIP
    and caveat is a human-readable note (empty string if none).
    """
    if result.relevance_score < RELEVANCE_THRESHOLD:
        return "SKIP", f"low relevance ({result.relevance_score:.0%})"

    if result.is_deprecated:
        return "WARN", "marked deprecated in GitLab docs"

    if len(result.text.strip()) < MIN_TEXT_LENGTH:
        return "WARN", "chunk is very short — may lack context"

    return "KEEP", ""


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _format_retrieval_results(
    results: list[RetrievalResult],
    query: str,
) -> str:
    """
    Format retrieval results for agents, with heuristic validation inline.
    SKIP chunks are dropped; WARN chunks are included with a caveat note.
    """
    kept = []
    skipped = 0

    for result in results:
        status, _ = _validate_chunk(result)
        if status != "SKIP":
            kept.append((result, status))
        else:
            skipped += 1

    if not kept:
        return (
            f"❌ No chunks met the minimum relevance threshold "
            f"({RELEVANCE_THRESHOLD:.0%}) for: {query!r}\n"
            f"Try rephrasing your question."
        )

    skip_note = f" ({skipped} low-relevance chunk(s) filtered out)" if skipped else ""
    output = f"📚 Found {len(kept)} validated chunks for: {query!r}{skip_note}\n\n"

    for i, (result, status) in enumerate(kept, 1):
        output += f"[{i}] 📄 {result.page_title}\n"
        output += f"    Section: {result.chunk_header}\n"
        output += f"    Relevance: {result.relevance_score:.0%}\n"
        output += f"    URL: {result.doc_url}\n"

        if result.gitlab_version_introduced:
            output += f"    Introduced in: GitLab {result.gitlab_version_introduced}\n"

        if result.is_deprecated:
            output += f"    ⚠️  DEPRECATED — include only if user asks about old versions\n"

        if result.has_code_example:
            output += f"    📝 Contains code example\n"

        if status == "WARN":
            _, caveat = _validate_chunk(result)
            output += f"    ⚠️  Caveat: {caveat}\n"

        output += f"\n    {result.text[:400]}\n"
        if len(result.text) > 400:
            output += f"    ... [truncated, see source]\n"

        output += "\n"

    return output


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# List of all tools available to agents
RETRIEVAL_TOOLS = [
    semantic_search,
    filtered_search,
    multi_query_search,
    get_chunk_by_id,
]
