"""
phase1_ingestion/vector_store.py
──────────────────────────────────
Manages the ChromaDB vector database — stores embeddings and retrieves
the most relevant chunks for a given query.

WHAT IS CHROMADB?
  ChromaDB is a local vector database. It runs entirely on your machine
  — no server, no account, no cost. It stores embeddings (lists of numbers)
  and finds the closest ones to a query embedding using cosine similarity.

HOW RETRIEVAL WORKS:
  1. User asks: "how do I cache node_modules in GitLab CI?"
  2. We convert that question into an embedding (list of numbers)
  3. ChromaDB finds the 10 chunks whose embeddings are most "similar"
  4. We return those chunks to the agent

COLLECTIONS:
  We use a single ChromaDB collection: "gitlab_docs"
  Each document in the collection = one DocChunk

FILTERING:
  ChromaDB supports metadata filtering — we can say:
  "Find relevant chunks, but ONLY from the 'ci' section"
  or "Only chunks where is_deprecated = False"
  This is what the Query Analyst Agent uses.

HOW TO READ THIS FILE:
  1. VectorStore.build(embedded_chunks) → stores all chunks in ChromaDB
  2. VectorStore.query() → searches for relevant chunks
  3. VectorStore.filtered_query() → searches with metadata filters
"""

import json
from pathlib import Path
from typing import Optional, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.table import Table

from config.settings import settings
from phase1_ingestion.chunker import DocChunk

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# RESULT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalResult:
    """
    A single retrieved chunk with its relevance score and metadata.
    This is what agents work with after retrieval.
    """

    def __init__(
        self,
        text: str,
        chunk_id: str,
        metadata: dict,
        distance: float,
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = metadata
        self.distance = distance
        # Convert distance to a 0-1 relevance score
        # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
        self.relevance_score = max(0.0, 1.0 - (distance / 2.0))

    @property
    def doc_url(self) -> str:
        return self.metadata.get("doc_url", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    @property
    def page_title(self) -> str:
        return self.metadata.get("page_title", "")

    @property
    def chunk_header(self) -> str:
        return self.metadata.get("chunk_header", "")

    @property
    def has_code_example(self) -> bool:
        return bool(self.metadata.get("has_code_example", False))

    @property
    def is_deprecated(self) -> bool:
        return bool(self.metadata.get("is_deprecated", False))

    @property
    def gitlab_version_introduced(self) -> str:
        return self.metadata.get("gitlab_version_introduced", "")

    def to_agent_context(self) -> str:
        """
        Format this result for injection into an agent's context.
        The agent reads this as part of its "retrieved knowledge".
        """
        lines = [
            f"📄 SOURCE: {self.page_title}",
            f"   Section: {self.chunk_header}",
            f"   URL: {self.doc_url}",
            f"   Relevance: {self.relevance_score:.0%}",
        ]
        if self.gitlab_version_introduced:
            lines.append(f"   Introduced in GitLab: {self.gitlab_version_introduced}")
        if self.is_deprecated:
            lines.append("   ⚠️  DEPRECATED")
        lines.extend(["", self.text, ""])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"score={self.relevance_score:.2f}, "
            f"section={self.section}, "
            f"title={self.page_title!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    ChromaDB-backed vector store for GitLab documentation.

    Provides:
    - build(): Ingest all embedded chunks
    - query(): Semantic search
    - filtered_query(): Semantic search with metadata filters
    - stats(): Collection statistics
    """

    UPSERT_BATCH_SIZE = 500   # ChromaDB performs well with 500-chunk batches

    def __init__(self):
        self.db_path = settings.chroma_db_path
        self.collection_name = settings.chroma_collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Connect to (or create) the ChromaDB database.
        ChromaDB stores data as files in self.db_path — no server needed.
        """
        self.db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # No telemetry
                allow_reset=True,
            )
        )

        # Get or create collection
        # We use cosine distance (best for semantic similarity of text)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )

        count = self._collection.count()
        logger.info(
            f"Connected to ChromaDB collection '{self.collection_name}' "
            f"({count:,} documents)"
        )

    def _ensure_connected(self) -> None:
        if self._collection is None:
            self.connect()

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD / INGEST
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        embedded_chunks: list[tuple[DocChunk, list[float]]],
        reset: bool = False,
    ) -> None:
        """
        Store all embedded chunks in ChromaDB.

        Args:
            embedded_chunks: List of (DocChunk, embedding_vector) pairs
            reset: If True, wipe existing data first (full re-index)
        """
        self._ensure_connected()

        console.print(
            "\n[bold cyan]🗄️  Phase 1 — Step 4: Building Vector Store[/bold cyan]\n"
        )

        if reset:
            console.print("[yellow]⚠️  Resetting collection (deleting existing data)...[/yellow]")
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        existing_count = self._collection.count()
        if existing_count > 0 and not reset:
            console.print(
                f"[yellow]⚡ Collection already has {existing_count:,} documents. "
                f"Upserting new/changed chunks only.[/yellow]\n"
            )

        # Prepare data in ChromaDB format
        ids, metadatas, documents, embeddings = [], [], [], []

        for chunk, embedding in embedded_chunks:
            chunk_id, metadata, text = chunk.to_chroma_format()
            ids.append(chunk_id)
            metadatas.append(metadata)
            documents.append(text)
            embeddings.append(embedding)

        # Upsert in batches (upsert = insert or update)
        total_batches = (len(ids) + self.UPSERT_BATCH_SIZE - 1) // self.UPSERT_BATCH_SIZE

        for i in track(
            range(0, len(ids), self.UPSERT_BATCH_SIZE),
            description=f"Storing {len(ids):,} chunks...",
            total=total_batches,
        ):
            batch_slice = slice(i, i + self.UPSERT_BATCH_SIZE)
            self._collection.upsert(
                ids=ids[batch_slice],
                embeddings=embeddings[batch_slice],
                metadatas=metadatas[batch_slice],
                documents=documents[batch_slice],
            )

        final_count = self._collection.count()
        console.print(f"\n[bold green]✅ Vector store ready![/bold green]")
        console.print(f"  Total documents: [bold]{final_count:,}[/bold]")
        console.print(f"  DB location: [dim]{self.db_path}[/dim]\n")

    # ─────────────────────────────────────────────────────────────────────────
    # QUERYING
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        exclude_deprecated: bool = True,
    ) -> list[RetrievalResult]:
        """
        Semantic search across ALL GitLab docs.

        Args:
            query_embedding: The embedded query vector
            n_results: Number of results to return
            exclude_deprecated: Filter out deprecated content

        Returns:
            List of RetrievalResult, sorted by relevance (most relevant first)
        """
        self._ensure_connected()

        where_filter = None
        if exclude_deprecated:
            where_filter = {"is_deprecated": {"$eq": False}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(results)

    def filtered_query(
        self,
        query_embedding: list[float],
        section: Optional[str] = None,
        subsection: Optional[str] = None,
        has_code: Optional[bool] = None,
        min_version: Optional[str] = None,
        exclude_deprecated: bool = True,
        n_results: int = 10,
    ) -> list[RetrievalResult]:
        """
        Semantic search with metadata filters.

        This is what the Query Analyst Agent uses when it knows the context:
        e.g., "search only in CI/CD docs" or "only chunks with code examples"

        Args:
            query_embedding: The embedded query vector
            section: Filter by top-level section ("ci", "api", "runner", etc.)
            subsection: Filter by subsection ("ci/yaml", "ci/docker", etc.)
            has_code: If True, only return chunks with code examples
            min_version: Only return chunks introduced in this version or later
            exclude_deprecated: Filter out deprecated content
            n_results: Number of results to return
        """
        self._ensure_connected()

        # Build ChromaDB $and filter
        conditions = []

        if exclude_deprecated:
            conditions.append({"is_deprecated": {"$eq": False}})
        if section:
            conditions.append({"section": {"$eq": section}})
        if subsection:
            conditions.append({"subsection": {"$eq": subsection}})
        if has_code is not None:
            conditions.append({"has_code_example": {"$eq": has_code}})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(results)

    def get_by_id(self, chunk_id: str) -> Optional[RetrievalResult]:
        """Retrieve a specific chunk by its ID (for citation lookup)."""
        self._ensure_connected()

        result = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        if not result["ids"]:
            return None

        return RetrievalResult(
            text=result["documents"][0],
            chunk_id=result["ids"][0],
            metadata=result["metadatas"][0],
            distance=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-QUERY FUSION (for hybrid retrieval)
    # ─────────────────────────────────────────────────────────────────────────

    def multi_query(
        self,
        query_embeddings: list[list[float]],
        n_results_per_query: int = 5,
        exclude_deprecated: bool = True,
    ) -> list[RetrievalResult]:
        """
        Run multiple query embeddings and fuse results using
        Reciprocal Rank Fusion (RRF).

        WHY: The Query Rewriter Agent generates 2-3 different phrasings
        of the user's question. Fusing all of them gives better recall
        than any single query alone.

        RRF formula: score(d) = sum(1 / (k + rank_i(d)))
        where k=60 is a constant that dampens the effect of high rankings.
        """
        self._ensure_connected()

        where_filter = {"is_deprecated": {"$eq": False}} if exclude_deprecated else None

        # Collect all ranked results
        all_ranked: list[list[RetrievalResult]] = []

        for embedding in query_embeddings:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results_per_query,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            all_ranked.append(self._parse_results(results))

        # Apply RRF
        return self._reciprocal_rank_fusion(all_ranked)

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[RetrievalResult]],
        k: int = 60,
    ) -> list[RetrievalResult]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.
        Returns a single merged list, deduplicated, sorted by fused score.
        """
        scores: dict[str, float] = {}
        results_by_id: dict[str, RetrievalResult] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                cid = result.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                if cid not in results_by_id:
                    results_by_id[cid] = result

        # Sort by fused score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = [results_by_id[cid] for cid in sorted_ids]

        return fused

    # ─────────────────────────────────────────────────────────────────────────
    # STATS & INSPECTION
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return collection statistics."""
        self._ensure_connected()
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "db_path": str(self.db_path),
        }

    def print_stats(self) -> None:
        """Print a rich table of collection statistics."""
        self._ensure_connected()

        table = Table(title="📊 Vector Store Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")

        s = self.stats()
        table.add_row("Collection Name", s["collection_name"])
        table.add_row("Total Chunks", f"{s['total_documents']:,}")
        table.add_row("Database Path", s["db_path"])
        table.add_row("Embedding Dimensions", "1536 (text-embedding-3-small)")
        table.add_row("Distance Metric", "Cosine")

        console.print(table)

    def test_query(self, query_text: str, embedder) -> None:
        """
        Quick sanity check: embed a test query and print top results.
        Call this after building the store to verify everything works.
        """
        console.print(f"\n[bold]🧪 Test Query:[/bold] {query_text!r}\n")

        embedding = embedder.embed_query(query_text)
        results = self.query(embedding, n_results=3)

        for i, result in enumerate(results, 1):
            console.print(
                f"  [{i}] [bold]{result.page_title}[/bold] — {result.chunk_header}"
            )
            console.print(
                f"      Score: [green]{result.relevance_score:.0%}[/green] | "
                f"URL: [dim]{result.doc_url}[/dim]"
            )
            console.print(f"      [dim]{result.text[:200]}...[/dim]\n")

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_results(self, raw: dict) -> list[RetrievalResult]:
        """Convert raw ChromaDB response into RetrievalResult objects."""
        results = []

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            results.append(
                RetrievalResult(
                    text=text,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    distance=distance,
                )
            )

        # Sort by distance (ascending = most relevant first)
        results.sort(key=lambda r: r.distance)
        return results
