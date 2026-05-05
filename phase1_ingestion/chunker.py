"""
phase1_ingestion/chunker.py
────────────────────────────
Transforms raw GitLab Markdown files into structured, metadata-rich chunks
ready for embedding.

WHY CHUNKING MATTERS:
  A vector database stores fixed-size text pieces (chunks).
  When a user asks a question, we retrieve the most relevant chunks.
  
  Bad chunking = bad retrieval = bad answers.
  
  We use MARKDOWN-AWARE chunking: we split on headers (## ###) 
  so each chunk stays within a logical section. This means a chunk 
  about "cache:key configuration" doesn't accidentally mix with 
  content about "artifacts".

METADATA WE ATTACH TO EVERY CHUNK:
  - source_file: which .md file it came from
  - section: top-level GitLab doc section (ci, api, runner, etc.)
  - subsection: nested folder (e.g., ci/yaml, ci/pipelines)
  - page_title: H1 title of the page
  - chunk_header: the H2/H3 header this chunk falls under
  - gitlab_version: extracted from "introduced in" tags if present
  - doc_url: reconstructed docs.gitlab.com URL
  - chunk_index: position of chunk within the page
  - token_count: approx tokens (for context window management)

HOW TO READ THIS FILE:
  1. MarkdownChunker.run() → processes all .md files → returns list of chunks
  2. _process_file() handles one file at a time
  3. _split_by_headers() does the actual splitting
  4. _extract_metadata() pulls version info, deprecation warnings, etc.
  5. Results are saved as JSON to data/processed/
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import tiktoken
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import track

from config.settings import settings

console = Console()

# Token counter (cl100k_base works for all OpenAI models)
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocChunk:
    """
    A single chunk of GitLab documentation ready to be embedded.
    This is the unit of retrieval — what gets stored in ChromaDB.
    """
    # Content
    text: str                        # The actual text to embed
    
    # Identity
    chunk_id: str                    # Unique ID: "ci_yaml_index_chunk_3"
    source_file: str                 # Relative path from /doc/
    doc_url: str                     # https://docs.gitlab.com/ee/...
    
    # Structure
    section: str                     # Top-level: "ci", "api", "runner"
    subsection: str                  # e.g., "ci/yaml", "ci/docker"
    page_title: str                  # H1 of the page
    chunk_header: str                # H2/H3 this chunk falls under
    chunk_index: int                 # Position in page (0, 1, 2...)
    
    # Version awareness (critical for GitLab — changes every month)
    gitlab_version_introduced: Optional[str] = None  # e.g., "16.0"
    gitlab_version_deprecated: Optional[str] = None  # e.g., "17.0"
    is_deprecated: bool = False
    
    # Quality signals
    token_count: int = 0
    has_code_example: bool = False   # Does this chunk contain YAML/code?
    has_warning: bool = False        # Does it contain a NOTE/WARNING block?
    
    # Extra metadata for filtering
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_chroma_format(self) -> tuple[str, dict, str]:
        """
        Returns (id, metadata, document) for ChromaDB insertion.
        ChromaDB metadata must be flat (no nested dicts).
        """
        metadata = {
            "source_file": self.source_file,
            "doc_url": self.doc_url,
            "section": self.section,
            "subsection": self.subsection,
            "page_title": self.page_title,
            "chunk_header": self.chunk_header,
            "chunk_index": self.chunk_index,
            "gitlab_version_introduced": self.gitlab_version_introduced or "",
            "gitlab_version_deprecated": self.gitlab_version_deprecated or "",
            "is_deprecated": self.is_deprecated,
            "token_count": self.token_count,
            "has_code_example": self.has_code_example,
            "has_warning": self.has_warning,
        }
        return self.chunk_id, metadata, self.text


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownChunker:
    """
    Converts raw GitLab Markdown files into DocChunks.
    
    Splitting strategy:
      1. Split on H2 headers (##) first — major sections
      2. If a section is still too large, split on H3 (###)
      3. If still too large, fall back to token-based splitting
    """

    # Regex patterns for GitLab-specific content
    FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
    VERSION_INTRO_RE = re.compile(
        r"(?:introduced|available|added)\s+in\s+gitlab\s+(\d+\.\d+)",
        re.IGNORECASE
    )
    VERSION_DEPRECATED_RE = re.compile(
        r"deprecated\s+in\s+gitlab\s+(\d+\.\d+)",
        re.IGNORECASE
    )
    H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
    H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)

    # GitLab docs base URL reconstructor
    # /doc/ci/yaml/index.md → https://docs.gitlab.com/ee/ci/yaml/
    DOCS_BASE_URL = "https://docs.gitlab.com/ee"

    def __init__(self):
        self.raw_path: Path = settings.gitlab_docs_raw_path
        self.processed_path: Path = settings.gitlab_docs_processed_path
        self.chunk_size: int = settings.chunk_size
        self.chunk_overlap: int = settings.chunk_overlap
        self.doc_path: Path = self.raw_path / "doc"

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> list[DocChunk]:
        """
        Process all markdown files and return list of DocChunks.
        Also saves chunks to data/processed/chunks.json for inspection.
        """
        console.print("\n[bold cyan]📄 Phase 1 — Step 2: Chunking Documents[/bold cyan]\n")

        if not self.doc_path.exists():
            raise FileNotFoundError(
                f"Doc path not found: {self.doc_path}\n"
                "Run the scraper first: python phase1_ingestion/run_ingestion.py"
            )

        md_files = self._discover_files()
        console.print(f"Found [bold]{len(md_files):,}[/bold] markdown files to process\n")

        all_chunks: list[DocChunk] = []

        for md_file in track(md_files, description="Chunking files..."):
            try:
                chunks = self._process_file(md_file)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Skipping {md_file.name}: {e}")
                continue

        # Filter out empty/useless chunks
        all_chunks = self._filter_chunks(all_chunks)

        # Save to disk for inspection / debugging
        self._save_chunks(all_chunks)

        console.print(f"\n[bold green]✅ Chunking complete![/bold green]")
        console.print(f"  Total chunks: [bold]{len(all_chunks):,}[/bold]")
        console.print(
            f"  Avg tokens/chunk: [bold]"
            f"{sum(c.token_count for c in all_chunks) // max(len(all_chunks), 1)}"
            f"[/bold]"
        )
        console.print(
            f"  Chunks with code: [bold]"
            f"{sum(1 for c in all_chunks if c.has_code_example):,}[/bold]"
        )
        console.print(
            f"  Deprecated content flagged: [bold]"
            f"{sum(1 for c in all_chunks if c.is_deprecated):,}[/bold]\n"
        )

        return all_chunks

    # ─────────────────────────────────────────────────────────────────────────
    # FILE DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def _discover_files(self) -> list[Path]:
        """Find all markdown files in the configured sections."""
        files = []
        for section in settings.gitlab_doc_sections:
            section_path = self.doc_path / section
            if section_path.exists():
                files.extend(section_path.rglob("*.md"))
            else:
                logger.debug(f"Section not found: {section_path}")
        
        # Also grab root-level docs
        for f in self.doc_path.glob("*.md"):
            files.append(f)

        # Sort for deterministic ordering
        return sorted(set(files))

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE FILE PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    def _process_file(self, md_file: Path) -> list[DocChunk]:
        """Process a single markdown file into chunks."""
        raw_text = md_file.read_text(encoding="utf-8", errors="ignore")

        if len(raw_text.strip()) < 100:
            return []  # Skip tiny files (redirects, stubs)

        # Extract YAML frontmatter
        frontmatter = self._parse_frontmatter(raw_text)
        content = self.FRONTMATTER_RE.sub("", raw_text).strip()

        # Extract page metadata
        page_title = self._extract_page_title(content, frontmatter, md_file)
        relative_path = str(md_file.relative_to(self.raw_path))  # doc/ci/yaml/index.md
        doc_url = self._build_url(md_file)
        section, subsection = self._extract_sections(md_file)

        # Split into sections by H2 headers
        sections = self._split_by_h2(content)

        chunks: list[DocChunk] = []
        chunk_index = 0

        for header, section_text in sections:
            # If section still too big, split further
            sub_texts = self._split_if_too_large(section_text)

            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue

                full_text = f"{header}\n\n{sub_text}".strip() if header else sub_text.strip()

                # Extract version info from this specific chunk
                version_intro = self._extract_version(
                    full_text, self.VERSION_INTRO_RE
                )
                version_deprecated = self._extract_version(
                    full_text, self.VERSION_DEPRECATED_RE
                )

                chunk = DocChunk(
                    text=full_text,
                    chunk_id=self._make_chunk_id(relative_path, chunk_index),
                    source_file=relative_path,
                    doc_url=doc_url,
                    section=section,
                    subsection=subsection,
                    page_title=page_title,
                    chunk_header=header or page_title,
                    chunk_index=chunk_index,
                    gitlab_version_introduced=version_intro,
                    gitlab_version_deprecated=version_deprecated,
                    is_deprecated=self._is_deprecated(full_text),
                    token_count=self._count_tokens(full_text),
                    has_code_example=self._has_code(full_text),
                    has_warning=self._has_warning(full_text),
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT SPLITTING
    # ─────────────────────────────────────────────────────────────────────────

    def _split_by_h2(self, text: str) -> list[tuple[str, str]]:
        """
        Split markdown text on H2 headers (##).
        Returns list of (header_text, section_content) pairs.
        """
        # Find all H2 positions
        h2_pattern = re.compile(r"^(##\s+.+)$", re.MULTILINE)
        matches = list(h2_pattern.finditer(text))

        if not matches:
            # No H2 headers — treat whole file as one section
            return [("", text)]

        sections = []

        # Content before first H2
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        # Each H2 section
        for i, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end() + 1
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((header, body))

        return sections

    def _split_if_too_large(self, text: str) -> list[str]:
        """
        If a section exceeds chunk_size tokens, split it further.
        First tries H3 splits, then falls back to sliding window.
        """
        if self._count_tokens(text) <= self.chunk_size:
            return [text]

        # Try H3 splits first
        h3_parts = self._split_by_h3(text)
        if len(h3_parts) > 1:
            # Some H3 parts might still be too large
            result = []
            for part in h3_parts:
                if self._count_tokens(part) <= self.chunk_size:
                    result.append(part)
                else:
                    result.extend(self._sliding_window_split(part))
            return result

        # Fall back to sliding window on paragraphs
        return self._sliding_window_split(text)

    def _split_by_h3(self, text: str) -> list[str]:
        """Split on H3 headers (###)."""
        h3_pattern = re.compile(r"^(###\s+.+)$", re.MULTILINE)
        matches = list(h3_pattern.finditer(text))

        if not matches:
            return [text]

        parts = []
        if text[:matches[0].start()].strip():
            parts.append(text[:matches[0].start()].strip())

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append(text[start:end].strip())

        return parts

    def _sliding_window_split(self, text: str) -> list[str]:
        """
        Last resort: split by paragraphs with overlap.
        Groups paragraphs until we hit chunk_size, then starts new chunk.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_paras = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_paras:
                chunks.append("\n\n".join(current_paras))
                
                # Overlap: keep last paragraph
                overlap_paras = current_paras[-1:] if self.chunk_overlap > 0 else []
                current_paras = overlap_paras + [para]
                current_tokens = sum(self._count_tokens(p) for p in current_paras)
            else:
                current_paras.append(para)
                current_tokens += para_tokens

        if current_paras:
            chunks.append("\n\n".join(current_paras))

        return chunks if chunks else [text]

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_frontmatter(self, text: str) -> dict:
        """Extract YAML frontmatter from markdown files."""
        match = self.FRONTMATTER_RE.match(text)
        if match:
            try:
                return yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                return {}
        return {}

    def _extract_page_title(
        self, content: str, frontmatter: dict, md_file: Path
    ) -> str:
        """Get page title from H1, frontmatter, or filename."""
        # Try frontmatter title first
        if "title" in frontmatter:
            return frontmatter["title"]
        # Try H1
        h1_match = self.H1_RE.search(content)
        if h1_match:
            return h1_match.group(1).strip()
        # Fall back to filename
        return md_file.stem.replace("-", " ").replace("_", " ").title()

    def _extract_sections(self, md_file: Path) -> tuple[str, str]:
        """
        Extract section and subsection from file path.
        doc/ci/yaml/index.md → ("ci", "ci/yaml")
        doc/runner/configuration/advanced-configuration.md → ("runner", "runner/configuration")
        """
        try:
            rel = md_file.relative_to(self.doc_path)
            parts = rel.parts
            section = parts[0] if parts else "other"
            subsection = "/".join(parts[:-1]) if len(parts) > 1 else section
            return section, subsection
        except ValueError:
            return "other", "other"

    def _build_url(self, md_file: Path) -> str:
        """
        Convert local file path to docs.gitlab.com URL.
        doc/ci/yaml/index.md → https://docs.gitlab.com/ee/ci/yaml/
        doc/ci/yaml/artifacts.md → https://docs.gitlab.com/ee/ci/yaml/artifacts/
        """
        try:
            rel = md_file.relative_to(self.doc_path)
            parts = list(rel.parts)
            # Remove 'index' from end — the URL doesn't include it
            if parts and parts[-1] in ("index.md", "index"):
                parts = parts[:-1]
            elif parts and parts[-1].endswith(".md"):
                parts[-1] = parts[-1][:-3]
            return self.DOCS_BASE_URL + "/" + "/".join(parts) + "/"
        except ValueError:
            return self.DOCS_BASE_URL

    def _extract_version(self, text: str, pattern: re.Pattern) -> Optional[str]:
        """Extract GitLab version number from text."""
        match = pattern.search(text)
        return match.group(1) if match else None

    def _is_deprecated(self, text: str) -> bool:
        """Check if this chunk contains a deprecation notice."""
        deprecation_signals = [
            "deprecated", "use instead", "scheduled for removal",
            "will be removed", "no longer supported"
        ]
        text_lower = text.lower()
        return any(signal in text_lower for signal in deprecation_signals)

    def _has_code(self, text: str) -> bool:
        """Check if chunk contains a code block."""
        return "```" in text

    def _has_warning(self, text: str) -> bool:
        """Check if chunk contains a NOTE/WARNING/DANGER block."""
        return bool(re.search(r">\s*(NOTE|WARNING|DANGER|CAUTION):", text, re.IGNORECASE))

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the cl100k_base tokenizer."""
        return len(TOKENIZER.encode(text))

    def _make_chunk_id(self, relative_path: str, index: int) -> str:
        """
        Create a deterministic, readable chunk ID.
        doc/ci/yaml/index.md + 3 → "ci__yaml__index__3"
        """
        clean = relative_path.replace("doc/", "").replace("/", "__").replace(".md", "")
        return f"{clean}__{index}"

    def _filter_chunks(self, chunks: list[DocChunk]) -> list[DocChunk]:
        """Remove low-quality chunks."""
        filtered = []
        for chunk in chunks:
            # Skip tiny chunks (navigation links, redirects)
            if chunk.token_count < 20:
                continue
            # Skip chunks that are pure code with no explanation
            lines = chunk.text.strip().split("\n")
            non_empty = [l for l in lines if l.strip()]
            if len(non_empty) < 3:
                continue
            filtered.append(chunk)
        
        removed = len(chunks) - len(filtered)
        if removed:
            logger.info(f"Filtered out {removed} low-quality chunks")
        
        return filtered

    def _save_chunks(self, chunks: list[DocChunk]) -> None:
        """Save all chunks to JSON for debugging and inspection."""
        self.processed_path.mkdir(parents=True, exist_ok=True)
        output_file = self.processed_path / "chunks.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in chunks],
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        console.print(f"  💾 Chunks saved to: [dim]{output_file}[/dim]")
