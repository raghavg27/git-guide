"""
config/settings.py
──────────────────
Central configuration module.
All settings loaded from environment variables (.env file).

COST BREAKDOWN (this setup):
  LLM calls    → OpenRouter free tier        = $0.00
  Embeddings   → Local sentence-transformers = $0.00
  Vector DB    → ChromaDB local files        = $0.00
  ─────────────────────────────────────────────────
  TOTAL COST   → $0.00 ✅

Import anywhere: from config.settings import settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseModel):

    # ── OpenRouter (Free LLM) ─────────────────────────────────────────────────
    openrouter_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL",
            "nvidia/nemotron-3-super-120b-a12b:free"
        )
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_name: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_APP_NAME", "GitLab-RAG-Chatbot"
        )
    )
    openrouter_app_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_APP_URL", "http://localhost:8501"
        )
    )

    # ── Local Embeddings (Free — runs on CPU) ─────────────────────────────────
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )
    # Vector size — must match the model chosen above:
    #   BAAI/bge-small-en-v1.5 → 384  (default)
    #   BAAI/bge-base-en-v1.5  → 768
    #   BAAI/bge-large-en-v1.5 → 1024
    #   all-MiniLM-L6-v2       → 384
    embedding_dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    gitlab_docs_raw_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / os.getenv(
            "GITLAB_DOCS_RAW_PATH", "data/raw/gitlab-docs"
        )
    )
    gitlab_docs_processed_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / os.getenv(
            "GITLAB_DOCS_PROCESSED_PATH", "data/processed"
        )
    )
    chroma_db_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / os.getenv(
            "CHROMA_DB_PATH", "vectorstore/chroma_db"
        )
    )
    log_file: Path = Field(
        default_factory=lambda: PROJECT_ROOT / os.getenv(
            "LOG_FILE", "logs/ingestion.log"
        )
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv(
            "CHROMA_COLLECTION_NAME", "gitlab_docs"
        )
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # ── GitLab Doc Sections ───────────────────────────────────────────────────
    gitlab_doc_sections: list[str] = Field(
        default=[
            "ci",
            "user",
            "administration",
            "api",
            "topics",
            "security",
            "runner",
            "update",
        ]
    )

    # ── LLM Generation Settings ───────────────────────────────────────────────
    llm_max_tokens: int = 2048
    # Low temperature = focused, factual answers (good for documentation)
    llm_temperature: float = 0.1

    def validate_setup(self) -> None:
        """Check required config is present. Raises clear error if not."""
        errors = []
        if not self.openrouter_api_key or "your-key-here" in self.openrouter_api_key:
            errors.append(
                "OPENROUTER_API_KEY is not set.\n"
                "    → Get a free key at: https://openrouter.ai/keys"
            )
        if errors:
            raise EnvironmentError(
                "\n\n❌ Configuration errors:\n" +
                "\n".join(f"  • {e}" for e in errors) +
                "\n\nFix these in your .env file (copy from .env.example)\n"
            )

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for d in [
            self.gitlab_docs_raw_path,
            self.gitlab_docs_processed_path,
            self.chroma_db_path,
            self.log_file.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get_llm_client(self):
        """
        Returns an OpenAI-compatible client pointed at OpenRouter.

        OpenRouter uses the exact same Python SDK as OpenAI.
        Only two things change: base_url and api_key.
        Model name is passed per-request (e.g. 'nvidia/nemotron-super-49b-v1:free')

        Usage:
            client = settings.get_llm_client()
            response = client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
            print(response.choices[0].message.content)
        """
        from openai import OpenAI
        return OpenAI(
            base_url=self.openrouter_base_url,
            api_key=self.openrouter_api_key,
            default_headers={
                "HTTP-Referer": self.openrouter_app_url,
                "X-Title": self.openrouter_app_name,
            },
        )

    class Config:
        arbitrary_types_allowed = True


# ── Singleton — import this everywhere ───────────────────────────────────────
settings = Settings()
