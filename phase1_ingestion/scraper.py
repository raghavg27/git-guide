"""
phase1_ingestion/scraper.py
────────────────────────────
Fetches GitLab documentation source files from GitLab's public repository.

STRATEGY:
  Primary   → Git sparse-checkout (gets clean Markdown, preserves structure)
  Fallback  → Direct HTTP download of individual doc pages

WHY SPARSE CHECKOUT?
  The full GitLab repo is ~4GB. The /doc folder alone is ~80MB.
  Sparse checkout lets us clone ONLY the /doc folder — fast and clean.

HOW TO READ THIS FILE:
  1. GitLabDocsScraper.run() is the main entry point
  2. It tries sparse_clone() first
  3. If that fails, it falls back to http_fallback()
  4. Results land in settings.gitlab_docs_raw_path
"""

import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from rich.console import Console
from rich.progress import track
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

console = Console()


class GitLabDocsScraper:
    """
    Downloads GitLab documentation source files.

    The GitLab docs live inside the main GitLab repo under /doc.
    Repo URL: https://gitlab.com/gitlab-org/gitlab
    """

    GITLAB_REPO_URL = "https://gitlab.com/gitlab-org/gitlab.git"
    RAW_BASE_URL = (
        "https://gitlab.com/gitlab-org/gitlab/-/raw/master/doc"
    )

    # Sitemap for fallback HTTP scraping
    DOCS_SITEMAP_URL = "https://docs.gitlab.com/sitemap.xml"

    def __init__(self):
        self.raw_path: Path = settings.gitlab_docs_raw_path
        self.doc_sections: list[str] = settings.gitlab_doc_sections

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """
        Main entry point. Returns path to downloaded docs.

        Steps:
          1. Check if docs already downloaded (skip if so)
          2. Try sparse git clone
          3. Fall back to HTTP if git fails
        """
        if self._docs_already_exist():
            console.print(
                f"[yellow]⚡ Docs already exist at {self.raw_path}. "
                f"Skipping download. Delete the folder to re-download.[/yellow]"
            )
            return self.raw_path

        console.print("\n[bold cyan]📥 Phase 1 — Step 1: Fetching GitLab Docs[/bold cyan]")
        console.print(f"Target path: [dim]{self.raw_path}[/dim]\n")

        success = self._sparse_clone()

        if not success:
            console.print(
                "[yellow]⚠️  Git sparse clone failed. "
                "Trying HTTP fallback...[/yellow]"
            )
            success = self._http_fallback()

        if not success:
            raise RuntimeError(
                "❌ Could not fetch GitLab docs via git or HTTP. "
                "Check your internet connection."
            )

        self._log_stats()
        return self.raw_path

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 1: GIT SPARSE CHECKOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _sparse_clone(self) -> bool:
        """
        Clone ONLY the /doc folder from GitLab's main repo.

        Git sparse-checkout means: "I want this repo, but only these folders."
        --filter=blob:none  → Don't download file contents yet (lazy)
        --no-checkout       → Don't write files yet
        Then we set sparse patterns and finally checkout only what we need.

        This takes ~2-5 minutes depending on your connection.
        """
        try:
            console.print("[bold]🔄 Attempting git sparse checkout...[/bold]")
            console.print(
                "[dim]This clones only the /doc folder (~80MB), "
                "not the full 4GB repo[/dim]\n"
            )

            target = self.raw_path
            target.mkdir(parents=True, exist_ok=True)

            # Step 1: Init empty repo
            self._run_git(["git", "init"], cwd=target)

            # Step 2: Add remote
            self._run_git(
                ["git", "remote", "add", "origin", self.GITLAB_REPO_URL],
                cwd=target
            )

            # Step 3: Enable sparse checkout
            self._run_git(
                ["git", "config", "core.sparseCheckout", "true"],
                cwd=target
            )

            # Step 4: Enable partial clone filter (don't download blobs yet)
            self._run_git(
                ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
                cwd=target
            )

            # Step 5: Write sparse-checkout patterns
            # We only want specific sections, not the entire /doc folder
            sparse_file = target / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)

            patterns = [f"doc/{section}/*" for section in self.doc_sections]
            patterns.append("doc/README.md")  # Root index

            sparse_file.write_text("\n".join(patterns))

            logger.info(f"Sparse checkout patterns: {patterns}")

            # Step 6: Fetch only the latest commit (depth=1), no history needed
            console.print("[dim]Fetching from GitLab (depth=1, no history)...[/dim]")
            self._run_git(
                [
                    "git", "fetch",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin", "master"
                ],
                cwd=target,
                timeout=300  # 5 min timeout
            )

            # Step 7: Checkout sparse files
            console.print("[dim]Checking out doc files...[/dim]")
            self._run_git(
                ["git", "checkout", "master"],
                cwd=target,
                timeout=300
            )

            # Verify we got files
            doc_path = target / "doc"
            if doc_path.exists() and any(doc_path.rglob("*.md")):
                md_count = len(list(doc_path.rglob("*.md")))
                console.print(
                    f"\n[bold green]✅ Sparse clone successful![/bold green] "
                    f"Got {md_count:,} markdown files."
                )
                return True
            else:
                logger.warning("Sparse clone ran but no .md files found")
                return False

        except Exception as e:
            logger.error(f"Sparse clone failed: {e}")
            # Clean up partial clone
            if self.raw_path.exists():
                shutil.rmtree(self.raw_path, ignore_errors=True)
            return False

    def _run_git(
        self,
        cmd: list[str],
        cwd: Path,
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """Run a git command, raise on failure."""
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd,
                output=result.stdout,
                stderr=result.stderr
            )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 2: HTTP FALLBACK (downloads key pages directly)
    # ─────────────────────────────────────────────────────────────────────────

    def _http_fallback(self) -> bool:
        """
        If git clone fails, download a curated set of critical doc pages
        directly from GitLab's raw file endpoint.

        This is slower but doesn't require git.
        We cover the most important docs for our CI/CD chatbot use case.
        """
        console.print("[bold]🌐 HTTP fallback: downloading key doc pages...[/bold]")

        # Curated list of the most important CI/CD docs
        # Format: (section, filename_without_extension)
        key_docs = [
            # CI/CD Core
            ("ci/yaml", "index"),
            ("ci/yaml", "artifacts"),
            ("ci/yaml", "needs"),
            ("ci/yaml", "rules"),
            ("ci/pipelines", "index"),
            ("ci/pipelines", "pipeline_architectures"),
            ("ci/pipelines", "merge_request_pipelines"),
            ("ci/caching", "index"),
            ("ci/runners", "index"),
            ("ci/runners", "configure_runners"),
            ("ci/docker", "using_docker_build"),
            ("ci/docker", "using_docker_images"),
            ("ci/variables", "index"),
            ("ci/variables", "predefined_variables"),
            ("ci/troubleshooting", "index"),
            ("ci/debugging", "index"),
            # Runner
            ("runner", "index"),
            ("runner/configuration", "advanced-configuration"),
            ("runner/executors", "docker"),
            ("runner/executors", "kubernetes"),
            # Merge Requests
            ("user/project/merge_requests", "index"),
            ("user/project/merge_requests", "approvals/index"),
            # Protected branches
            ("user/project/protected_branches", "index"),
            # Permissions
            ("user/permissions", "index"),
            # API
            ("api", "index"),
            ("api", "rest/index"),
        ]

        self.raw_path.mkdir(parents=True, exist_ok=True)
        success_count = 0

        for section, filename in track(key_docs, description="Downloading docs..."):
            url = f"{self.RAW_BASE_URL}/{section}/{filename}.md"
            local_path = self.raw_path / "doc" / section / f"{filename}.md"
            local_path.parent.mkdir(parents=True, exist_ok=True)

            content = self._fetch_raw_file(url)
            if content:
                local_path.write_text(content, encoding="utf-8")
                success_count += 1
            else:
                logger.warning(f"Could not fetch: {url}")

        console.print(
            f"\n[bold green]✅ HTTP fallback: downloaded {success_count}/{len(key_docs)} docs[/bold green]"
        )
        return success_count > 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _fetch_raw_file(self, url: str) -> Optional[str]:
        """Download a single raw file with retry logic."""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            return None
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            raise  # Let tenacity retry

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _docs_already_exist(self) -> bool:
        """Check if docs were already downloaded."""
        doc_path = self.raw_path / "doc"
        return doc_path.exists() and any(doc_path.rglob("*.md"))

    def _log_stats(self) -> None:
        """Print a summary of what was downloaded."""
        doc_path = self.raw_path / "doc"
        md_files = list(doc_path.rglob("*.md"))

        # Count by section
        section_counts: dict[str, int] = {}
        for f in md_files:
            # Get the first folder level under /doc
            try:
                rel = f.relative_to(doc_path)
                section = rel.parts[0] if len(rel.parts) > 1 else "root"
                section_counts[section] = section_counts.get(section, 0) + 1
            except ValueError:
                pass

        console.print("\n[bold]📊 Downloaded docs breakdown:[/bold]")
        for section, count in sorted(section_counts.items()):
            console.print(f"  [cyan]{section:20s}[/cyan] {count:4d} files")
        console.print(f"\n  [bold]Total: {len(md_files):,} markdown files[/bold]")
        console.print(
            f"\n  📁 Saved to: [dim]{self.raw_path}[/dim]\n"
        )
