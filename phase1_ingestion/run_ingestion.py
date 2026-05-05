"""
phase1_ingestion/run_ingestion.py
──────────────────────────────────
The MAIN SCRIPT for Phase 1. Run this once to build your knowledge base.

This script orchestrates the full pipeline:

  Step 1 — SCRAPE   → Clone GitLab docs from GitLab's public repo
  Step 2 — CHUNK    → Split docs into smart, metadata-rich chunks
  Step 3 — EMBED    → Convert chunks to vector embeddings (OpenAI API)
  Step 4 — STORE    → Save embeddings to ChromaDB (local vector DB)
  Step 5 — VERIFY   → Run test queries to confirm everything works

HOW TO RUN:
  # First time (full pipeline):
  python -m phase1_ingestion.run_ingestion

  # Reset and re-index everything:
  python -m phase1_ingestion.run_ingestion --reset

  # Skip scraping (docs already downloaded):
  python -m phase1_ingestion.run_ingestion --skip-scrape

  # Just verify an existing store:
  python -m phase1_ingestion.run_ingestion --verify-only

ESTIMATED TIME (first run):
  Scraping:   3-8 min  (sparse git clone)
  Chunking:   1-2 min  (~8,000 files)
  Embedding:  5-15 min (~30,000 chunks, depends on API speed)
  Storing:    1-2 min  (ChromaDB upsert)
  Total:      ~15-30 minutes, ~$0.05 cost

SUBSEQUENT RUNS:
  Much faster — scraping is skipped (docs exist),
  embedding uses cache (only new/changed chunks embedded).
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# ── Add project root to sys.path so imports work ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from phase1_ingestion.scraper import GitLabDocsScraper
from phase1_ingestion.chunker import MarkdownChunker
from phase1_ingestion.embedder import DocEmbedder
from phase1_ingestion.vector_store import VectorStore

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full Phase 1 data ingestion pipeline.
    Each step is independent — failures are isolated and reported clearly.
    """

    def __init__(self):
        self.scraper = GitLabDocsScraper()
        self.chunker = MarkdownChunker()
        self.embedder = DocEmbedder()
        self.vector_store = VectorStore()

    def run(
        self,
        skip_scrape: bool = False,
        reset: bool = False,
        verify_only: bool = False,
    ) -> None:
        """
        Execute the full ingestion pipeline.

        Args:
            skip_scrape: Skip git clone (use existing docs)
            reset: Wipe ChromaDB and re-index from scratch
            verify_only: Only run verification queries (skip ingestion)
        """
        start_time = time.time()
        self._print_header()

        # ── Validate config ────────────────────────────────────────────────
        try:
            settings.validate_setup()
            settings.ensure_directories()
            console.print(
                f"[green]✅ Configuration validated[/green] | "
                f"LLM: [cyan]{settings.openrouter_model}[/cyan] via OpenRouter | "
                f"Embeddings: [cyan]{settings.embedding_model}[/cyan] (local)\n"
            )
        except EnvironmentError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

        # ── Verify only mode ──────────────────────────────────────────────
        if verify_only:
            self._run_verification()
            return

        # ── Step 1: Scrape ─────────────────────────────────────────────────
        console.print(Rule("[bold]Step 1 of 5 — Fetch GitLab Docs[/bold]"))
        if skip_scrape:
            console.print(
                "[yellow]⏩ Skipping scrape (--skip-scrape flag)[/yellow]\n"
            )
        else:
            try:
                self.scraper.run()
            except Exception as e:
                console.print(f"[red]❌ Scraping failed: {e}[/red]")
                sys.exit(1)

        # ── Step 2: Chunk ──────────────────────────────────────────────────
        console.print(Rule("[bold]Step 2 of 5 — Chunk Documents[/bold]"))
        try:
            chunks = self.chunker.run()
        except Exception as e:
            console.print(f"[red]❌ Chunking failed: {e}[/red]")
            sys.exit(1)

        if not chunks:
            console.print(
                "[red]❌ No chunks produced. "
                "Check that docs were downloaded correctly.[/red]"
            )
            sys.exit(1)

        # ── Step 3: Embed ──────────────────────────────────────────────────
        console.print(Rule("[bold]Step 3 of 5 — Generate Embeddings[/bold]"))
        try:
            embedded_chunks = self.embedder.run(chunks)
        except Exception as e:
            console.print(f"[red]❌ Embedding failed: {e}[/red]")
            console.print(
                "[dim]Check your OPENAI_API_KEY in .env "
                "and verify you have API credits.[/dim]"
            )
            sys.exit(1)

        # ── Step 4: Store ──────────────────────────────────────────────────
        console.print(Rule("[bold]Step 4 of 5 — Store in ChromaDB[/bold]"))
        try:
            self.vector_store.build(embedded_chunks, reset=reset)
        except Exception as e:
            console.print(f"[red]❌ Vector store build failed: {e}[/red]")
            sys.exit(1)

        # ── Step 5: Verify ─────────────────────────────────────────────────
        console.print(Rule("[bold]Step 5 of 5 — Verification[/bold]"))
        self._run_verification()

        # ── Summary ────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        self._print_summary(chunks, embedded_chunks, elapsed)

    # ─────────────────────────────────────────────────────────────────────────
    # VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def _run_verification(self) -> None:
        """
        Run sample queries against the vector store.
        This proves the pipeline worked end-to-end.
        These test queries cover our main chatbot use cases.
        """
        console.print("\n[bold]🧪 Running verification queries...[/bold]\n")

        test_queries = [
            # Core CI/CD debugging queries
            "how to fix exit code 137 in GitLab CI pipeline",
            "cache node_modules between pipeline jobs",
            "rules:when syntax error in gitlab-ci.yml",
            # Runner
            "configure Docker executor for GitLab runner",
            # API
            "list open merge requests using GitLab REST API",
            # Permissions
            "difference between developer and maintainer role in GitLab",
        ]

        self.vector_store.connect()

        all_passed = True
        for query in test_queries:
            try:
                embedding = self.embedder.embed_query(query)
                results = self.vector_store.query(embedding, n_results=3)

                if results:
                    top = results[0]
                    status = "[green]✅[/green]"
                    detail = (
                        f"{top.page_title!r} "
                        f"(score: {top.relevance_score:.0%})"
                    )
                else:
                    status = "[yellow]⚠️ [/yellow]"
                    detail = "No results returned"
                    all_passed = False

                console.print(f"  {status} Query: [cyan]{query!r}[/cyan]")
                console.print(f"       → {detail}\n")

            except Exception as e:
                console.print(
                    f"  [red]❌[/red] Query failed: {query!r}\n"
                    f"       Error: {e}\n"
                )
                all_passed = False

        if all_passed:
            console.print(
                "[bold green]✅ All verification queries passed![/bold green]\n"
            )
        else:
            console.print(
                "[yellow]⚠️  Some queries returned no results. "
                "This may be normal if only partial docs were downloaded.[/yellow]\n"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLAY HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _print_header(self) -> None:
        console.print(
            Panel.fit(
                "[bold cyan]🦊 GitLab Agentic RAG Chatbot[/bold cyan]\n"
                "[white]Phase 1 — Data Ingestion Pipeline[/white]\n\n"
                "[dim]GitLab Docs → Chunks → Local Embeddings → ChromaDB[/dim]\n"
                "[bold green]💰 Total cost: $0.00[/bold green]",
                border_style="cyan",
                padding=(1, 4),
            )
        )
        console.print()

    def _print_summary(
        self,
        chunks,
        embedded_chunks,
        elapsed: float,
    ) -> None:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        console.print(
            Panel.fit(
                f"[bold green]🎉 Phase 1 Complete![/bold green]\n\n"
                f"  📄 Chunks created:    [bold]{len(chunks):,}[/bold]\n"
                f"  🔢 Vectors stored:    [bold]{len(embedded_chunks):,}[/bold]\n"
                f"  🗄️  DB location:       [dim]{settings.chroma_db_path}[/dim]\n"
                f"  ⏱️  Total time:        [bold]{minutes}m {seconds}s[/bold]\n\n"
                f"[dim]Next step: Phase 2 — Build CrewAI Agents[/dim]\n"
                f"[dim]Run: python -m phase2_agents.run_agents[/dim]",
                border_style="green",
                padding=(1, 4),
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GitLab RAG — Phase 1: Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (first time):
  python -m phase1_ingestion.run_ingestion

  # Re-index everything from scratch:
  python -m phase1_ingestion.run_ingestion --reset

  # Skip git clone (docs already downloaded):
  python -m phase1_ingestion.run_ingestion --skip-scrape

  # Just verify an existing vector store:
  python -m phase1_ingestion.run_ingestion --verify-only
        """
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ChromaDB and re-index from scratch",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip git clone step (use already-downloaded docs)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification queries against existing store",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(sys.stderr, level="WARNING")  # Only warnings to console

    pipeline = IngestionPipeline()
    pipeline.run(
        skip_scrape=args.skip_scrape,
        reset=args.reset,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    main()
