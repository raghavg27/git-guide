"""
phase2_agents/run_agents.py
───────────────────────────
Main script to test Phase 2 — the CrewAI agents.

This script:
  1. Creates the crew (all agents)
  2. Runs test queries
  3. Shows how the agents collaborate
  4. Displays final answers with full citations

RUN THIS:
  python -m phase2_agents.run_agents

OPTIONAL FLAGS:
  --simple       Use simplified crew (just Retriever + Synthesiser)
  --test-query   Run a specific query instead of interactive mode
  --verbose      Show all agent reasoning (very detailed)
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from config.settings import settings
from config.llm_client import test_connection
from phase2_agents.crew import create_simple_retrieval_crew
from phase2_agents.parallel_pipeline import run_parallel_preprocess

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# QUERY ROUTING — decides whether to run the full 4-agent pipeline or the
# lightweight 2-agent pipeline (Retriever + Synthesiser only).
#
# The Intent Classifier + Query Rewriter add ~6-10s of LLM calls but only
# help when the query is vague, short, or missing domain terminology.
# Clear, well-formed questions don't need them.
# ─────────────────────────────────────────────────────────────────────────────

_GITLAB_KEYWORDS = {
    "pipeline", "ci", "cd", "runner", "cache", "artifact", "job", "stage",
    "merge", "request", "branch", "commit", "deploy", "kubernetes", "docker",
    "yaml", "api", "token", "permission", "role", "group", "registry",
    "variable", "trigger", "schedule", "webhook", "sast", "dast", "scan",
    "gitlab", "gitops", "helm", "terraform", "secret", "environment",
    "executor", "tag", "protected", "approval", "review", "issue",
}

_QUESTION_STARTERS = {
    "how", "what", "why", "when", "where", "which", "can", "does",
    "is", "are", "will", "should", "would",
}


def _needs_full_pipeline(query: str) -> bool:
    """
    True  → run Intent Classifier + Query Rewriter before retrieval.
    False → skip straight to Retriever + Synthesiser.

    Rules (in order):
      < 4 words                              → True  (too vague to retrieve well)
      starts with question word + 5+ words  → False (well-formed question)
      has a GitLab keyword + 6+ words       → False (specific enough)
      everything else                        → True  (rewriting may help)
    """
    words = query.lower().split()
    n = len(words)

    if n < 4:
        return True

    if words[0] in _QUESTION_STARTERS and n >= 5:
        return False

    if any(w in _GITLAB_KEYWORDS for w in words) and n >= 6:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="GitLab RAG — Phase 2: CrewAI Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (ask questions)
  python -m phase2_agents.run_agents

  # Test a single query
  python -m phase2_agents.run_agents --test-query "how do I cache in gitlab ci?"

  # Use simplified crew (faster, less reasoning)
  python -m phase2_agents.run_agents --simple

  # Verbose output (see all agent reasoning)
  python -m phase2_agents.run_agents --verbose
        """
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simplified crew (Retriever + Synthesiser only)",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed agent reasoning",
    )

    args = parser.parse_args()

    _print_header()

    # Validate setup
    try:
        settings.validate_setup()
        console.print("[green]✅ Configuration validated[/green]\n")
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Test OpenRouter connection
    console.print("[bold]Testing OpenRouter connection...[/bold]")
    if not test_connection():
        console.print(
            "\n[red]❌ OpenRouter connection failed.[/red]\n"
            "Check your OPENROUTER_API_KEY in .env"
        )
        sys.exit(1)

    console.print(Rule("[bold]Creating Crew[/bold]"))

    simple_crew = create_simple_retrieval_crew()

    if args.simple:
        console.print(
            "Using [yellow]simple pipeline[/yellow] (Retrieve → Synthesise)\n"
        )
    else:
        console.print(
            "Using [cyan]auto-routing[/cyan] — "
            "clear questions go straight to retrieval, "
            "vague/short ones get intent+rewrite in parallel first\n"
        )

    # Single query mode
    if args.test_query:
        console.print(Rule("[bold]Running Query[/bold]"))
        _run_single_query(simple_crew, args.test_query, force_simple=args.simple)
        return

    # Interactive mode
    console.print(Rule("[bold]Interactive Mode[/bold]"))
    console.print("[dim]Type 'exit' to quit[/dim]\n")

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if query.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not query:
                continue

            _run_single_query(simple_crew, query, force_simple=args.simple)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            break


def _run_single_query(simple_crew, query: str, force_simple: bool = False) -> None:
    """Route the query to the appropriate pipeline and execute it."""
    console.print(f"\n[bold cyan]Question:[/bold cyan] {query}\n")

    if not force_simple and _needs_full_pipeline(query):
        console.print(
            "[dim]Pipeline: full — intent+rewrite in parallel → retrieve → synthesise[/dim]"
        )
        preprocess_context = asyncio.run(run_parallel_preprocess(query))
        enriched_query = f"{query}\n\n[PREPROCESSING CONTEXT]\n{preprocess_context}"
    else:
        console.print("[dim]Pipeline: simple (retrieve → synthesise)[/dim]")
        enriched_query = query

    try:
        result = simple_crew.kickoff(inputs={"user_query": enriched_query})

        console.print(Rule("[bold green]Agent Collaboration Complete[/bold green]"))
        console.print()

        if result:
            console.print("[bold]Answer:[/bold]\n")
            console.print(result)
        else:
            console.print("[yellow]No response generated[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {e}")
        console.print("\n[dim]Enable --verbose for more details[/dim]")


def _print_header() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]🦊 GitLab Agentic RAG Chatbot[/bold cyan]\n"
            "[white]Phase 2 — CrewAI Agents[/white]\n\n"
            "[dim]Auto-routing: simple queries → Retrieve → Synthesise\n"
            "Vague queries → Intent → Rewrite → Retrieve → Synthesise[/dim]\n"
            "[bold green]💰 Total cost: $0.00[/bold green]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


if __name__ == "__main__":
    main()
