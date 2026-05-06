"""
phase2_agents/crew.py
──────────────────────
CrewAI Crew + Task definitions.

PIPELINE OVERVIEW:

  Simple queries (clear, well-formed):
    Retrieve → Synthesise
    (create_simple_retrieval_crew)

  Vague / short queries:
    [async parallel] Intent Classify + Query Rewrite  ← parallel_pipeline.py
          ↓ combined context injected into query
    Retrieve → Synthesise
    (still uses create_simple_retrieval_crew, with enriched input)

  Heuristic validation (no LLM) runs inside _format_retrieval_results()
  in retrieval_tools.py — chunks below 0.40 relevance are dropped before
  the synthesiser sees them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from crewai import Crew, Task, Process

from phase2_agents.agents.retriever import create_retriever_agent
from phase2_agents.agents.synthesiser import create_synthesiser_agent


def create_simple_retrieval_crew() -> Crew:
    """
    2-agent crew: Retriever → Synthesiser.

    Used for all queries. Vague queries get their input pre-enriched by
    parallel_pipeline.run_parallel_preprocess() before kickoff is called.
    """

    retriever   = create_retriever_agent()
    synthesiser = create_synthesiser_agent()

    task_retrieve = Task(
        description=(
            "Search GitLab documentation for:\n\n"
            "QUESTION: {user_query}\n\n"
            "Use semantic_search or filtered_search to find the 5 most "
            "relevant documentation chunks. Prefer chunks with code examples.\n\n"
            "If the question contains a [PREPROCESSING CONTEXT] block with intent "
            "and rewrite suggestions, use those to guide your search strategy and "
            "apply any recommended section filters."
        ),
        expected_output=(
            "Top 5 documentation chunks with page title, URL, relevance score, "
            "and full text of each chunk."
        ),
        agent=retriever,
    )

    task_synthesise = Task(
        description=(
            "Write a clear, cited answer using the retrieved chunks.\n\n"
            "QUESTION: {user_query}\n\n"
            "Include: direct answer, explanation with citations, code example "
            "if applicable, any caveats. Cite source URLs for every claim."
        ),
        expected_output=(
            "Complete answer with citations, code examples if applicable, "
            "and source URLs for all technical claims."
        ),
        agent=synthesiser,
        context=[task_retrieve],
    )

    crew = Crew(
        agents=[retriever, synthesiser],
        tasks=[task_retrieve, task_synthesise],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    return crew
