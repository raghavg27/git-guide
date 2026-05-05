"""
phase1_ingestion/embedder.py
─────────────────────────────
Converts DocChunks into vector embeddings using a LOCAL model.

💰 COST: $0.00 — runs entirely on your CPU. No API. No internet after
         the first download.

MODEL: BAAI/bge-small-en-v1.5
  - Downloaded once from HuggingFace (~90MB) and cached locally
  - 384-dimensional embeddings
  - Excellent quality for technical English text
  - Fast on CPU (no GPU needed)
  - Industry-standard for RAG systems

WHY BGE (BAAI General Embedding)?
  BGE models are specifically trained for retrieval tasks.
  They outperform OpenAI ada-002 on many benchmarks while being free.
  "bge-small" is the fastest variant — perfect for development.
  Switch to "bge-large" in production for better quality.

BATCHING:
  We embed chunks in batches of 64 to manage RAM usage.
  Larger batches = faster, but more RAM. 64 is safe for 8GB RAM.

HOW TO READ THIS FILE:
  1. DocEmbedder.__init__() loads the model (first call downloads it)
  2. DocEmbedder.run(chunks) → embeds all chunks → returns results
  3. DocEmbedder.embed_query(text) → embeds a single query at runtime
  4. Results cached to disk so re-runs don't re-embed everything
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TaskProgressColumn, TimeRemainingColumn
)
from sentence_transformers import SentenceTransformer

from config.settings import settings
from phase1_ingestion.chunker import DocChunk

console = Console()


class DocEmbedder:
    """
    Embeds DocChunks using a local sentence-transformers model.

    The model is downloaded from HuggingFace on first use and cached
    at ~/.cache/huggingface/ — subsequent runs load from cache instantly.

    Key features:
      - Completely free (no API)
      - Works offline after first download
      - Batched processing with progress display
      - Disk cache — only embeds new/changed chunks on re-runs
      - BGE models use a query prefix for better retrieval accuracy
    """

    BATCH_SIZE = 64     # Chunks per forward pass through the model
                        # Reduce to 32 if you run out of RAM

    # BGE models need a special prefix on queries (not on documents)
    # This is a BGE-specific trick that improves retrieval accuracy
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        self.model_name = settings.embedding_model
        self.processed_path = settings.gitlab_docs_processed_path
        self.cache_file = self.processed_path / "embedding_cache.json"
        self._model: Optional[SentenceTransformer] = None
        self._cache: dict[str, list[float]] = self._load_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        """
        Load the embedding model.
        First call downloads from HuggingFace (~90MB for bge-small).
        Subsequent calls load from local cache in ~/.cache/huggingface/
        """
        if self._model is None:
            console.print(
                f"\n[bold]📦 Loading embedding model:[/bold] "
                f"[cyan]{self.model_name}[/cyan]"
            )
            console.print(
                "[dim]First run: downloads ~90MB from HuggingFace. "
                "Subsequent runs: loads from local cache instantly.[/dim]\n"
            )

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",        # CPU only — no GPU needed
            )

            dim = self._model.get_sentence_embedding_dimension()
            console.print(
                f"[green]✅ Model loaded[/green] | "
                f"Dimensions: [bold]{dim}[/bold] | "
                f"Device: CPU\n"
            )

        return self._model

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self, chunks: list[DocChunk]
    ) -> list[tuple[DocChunk, list[float]]]:
        """
        Embed all chunks. Returns list of (chunk, embedding_vector) pairs.

        Uses disk cache: chunks already embedded in a previous run are
        skipped. This means you can safely interrupt and resume without
        losing progress.
        """
        console.print(
            "\n[bold cyan]🔢 Phase 1 — Step 3: Generating Embeddings[/bold cyan]"
        )
        console.print(
            f"Model: [bold]{self.model_name}[/bold] | "
            f"Cost: [bold green]$0.00[/bold green] | "
            f"Runs locally on CPU\n"
        )

        model = self._load_model()

        # Separate cached vs uncached
        to_embed: list[DocChunk] = []
        cached_results: list[tuple[DocChunk, list[float]]] = []

        for chunk in chunks:
            key = self._cache_key(chunk.text)
            if key in self._cache:
                cached_results.append((chunk, self._cache[key]))
            else:
                to_embed.append(chunk)

        if cached_results:
            console.print(
                f"  ⚡ [yellow]{len(cached_results):,} chunks from cache[/yellow]"
            )
        if to_embed:
            console.print(
                f"  🔄 [cyan]{len(to_embed):,} chunks to embed[/cyan]"
            )

        console.print()

        if not to_embed:
            console.print(
                "[green]✅ All chunks already cached! Skipping embedding.[/green]\n"
            )
            return cached_results

        # Estimate time
        est_seconds = (len(to_embed) / self.BATCH_SIZE) * 2
        console.print(
            f"  ⏱️  Estimated time: ~{int(est_seconds // 60)}m "
            f"{int(est_seconds % 60)}s on CPU\n"
        )

        # Embed in batches
        new_results = self._embed_all_batches(model, to_embed)

        # Merge with cached
        all_results = cached_results + new_results

        # Persist cache
        self._save_cache()

        console.print(f"\n[bold green]✅ Embedding complete![/bold green]")
        console.print(f"  Total vectors: [bold]{len(all_results):,}[/bold]")
        console.print(f"  Total cost:    [bold green]$0.00[/bold green]\n")

        return all_results

    # ─────────────────────────────────────────────────────────────────────────
    # BATCHED EMBEDDING
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_all_batches(
        self,
        model: SentenceTransformer,
        chunks: list[DocChunk],
    ) -> list[tuple[DocChunk, list[float]]]:
        """Embed chunks in batches, showing a progress bar."""

        results: list[tuple[DocChunk, list[float]]] = []
        batches = [
            chunks[i: i + self.BATCH_SIZE]
            for i in range(0, len(chunks), self.BATCH_SIZE)
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Embedding {len(chunks):,} chunks locally...",
                total=len(batches),
            )

            for batch in batches:
                texts = [c.text for c in batch]

                # sentence-transformers handles batching internally
                # normalize_embeddings=True → better cosine similarity
                vectors: np.ndarray = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                for chunk, vector in zip(batch, vectors):
                    embedding = vector.tolist()
                    results.append((chunk, embedding))
                    # Cache it
                    self._cache[self._cache_key(chunk.text)] = embedding

                progress.advance(task)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY EMBEDDING (called at retrieval time by agents)
    # ─────────────────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        BGE models use a special prefix on queries (but NOT on documents).
        This asymmetric approach improves retrieval accuracy significantly.

        Called by Retriever Agent every time a user asks a question.
        Takes ~0.1 seconds on CPU — fast enough for real-time use.
        """
        model = self._load_model()

        # Add BGE query prefix if using a BGE model
        if "bge" in self.model_name.lower():
            query_text = self.BGE_QUERY_PREFIX + query
        else:
            query_text = query

        vector: np.ndarray = model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector[0].tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """
        Embed multiple query strings at once.
        Used by multi-query retrieval (Query Rewriter Agent generates
        2-3 phrasings, we embed all of them for better recall).
        """
        model = self._load_model()

        if "bge" in self.model_name.lower():
            texts = [self.BGE_QUERY_PREFIX + q for q in queries]
        else:
            texts = queries

        vectors: np.ndarray = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        """MD5 of text content — same text always gets same key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, list[float]]:
        """Load existing embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache):,} cached embeddings")
                return cache
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Starting fresh.")
        return {}

    def _save_cache(self) -> None:
        """Persist embedding cache to disk."""
        self.processed_path.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f)
        logger.info(f"Saved {len(self._cache):,} embeddings to cache")
        console.print(
            f"  💾 Cache saved: [dim]{self.cache_file}[/dim] "
            f"({len(self._cache):,} entries)"
        )
