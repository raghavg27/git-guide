"""
config/llm_client.py
──────────────────────
Central LLM client for all CrewAI agents — powered by OpenRouter.

WHY A WRAPPER?
  All agents share one LLM client. Centralising it here means:
  - One place to swap models (change .env → affects all agents)
  - One place to add retry logic, logging, cost tracking
  - Easy to test with mock client

HOW OPENROUTER WORKS:
  OpenRouter is a router — it sits between you and 100+ LLM providers
  (OpenAI, Anthropic, Google, NVIDIA, Meta, Mistral...).
  Free models are truly free — OpenRouter subsidises them.
  You just need a free account and API key.

  Technically: OpenRouter exposes an OpenAI-compatible REST API.
  So we use the openai Python SDK — just with a different base_url.

FREE MODELS AVAILABLE (as of 2025):
  nvidia/nemotron-super-49b-v1:free   ← powerful, great reasoning
  meta-llama/llama-3.3-70b-instruct:free
  google/gemma-3-27b-it:free
  mistralai/mistral-7b-instruct:free  ← fastest
  deepseek/deepseek-r1:free           ← strong at technical content

HOW TO READ THIS FILE:
  1. get_llm_response() → simple one-shot call to the LLM
  2. get_llm_client()   → returns raw OpenAI client for advanced use
  3. test_connection()  → sanity check your OpenRouter setup
"""

from typing import Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE CALL (used by most agents)
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=15),
)
def get_llm_response(
    prompt: str,
    system_prompt: str = "You are a helpful GitLab documentation expert.",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Send a prompt to OpenRouter and return the text response.

    Retried up to 3 times on failure (handles rate limits / timeouts).

    Args:
        prompt: The user message / task
        system_prompt: The agent's persona / instructions
        temperature: Override default temperature (0.0–1.0)
        max_tokens: Override default max output tokens

    Returns:
        The model's response as a plain string

    Example:
        response = get_llm_response(
            prompt="Explain GitLab CI cache configuration",
            system_prompt="You are a GitLab CI/CD expert."
        )
    """
    client = settings.get_llm_client()

    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )

    text = response.choices[0].message.content or ""
    logger.debug(
        f"LLM response: {len(text)} chars | "
        f"model={settings.openrouter_model}"
    )
    return text


def get_llm_response_with_history(
    messages: list[dict],
    system_prompt: str = "You are a helpful GitLab documentation expert.",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Send a full conversation history to the LLM.
    Used by the chat interface for multi-turn conversations.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
        system_prompt: The agent persona

    Example:
        history = [
            {"role": "user", "content": "What is GitLab CI?"},
            {"role": "assistant", "content": "GitLab CI is..."},
            {"role": "user", "content": "How do I configure a runner?"},
        ]
        response = get_llm_response_with_history(history)
    """
    client = settings.get_llm_client()

    full_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]

    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=full_messages,
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )

    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# RAW CLIENT (for advanced use in CrewAI agents)
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_client():
    """Return the raw OpenAI-compatible client for OpenRouter."""
    return settings.get_llm_client()


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_connection() -> bool:
    """
    Test that OpenRouter is configured correctly.
    Sends a minimal request and checks for a valid response.

    Run this directly to verify your setup:
        python -c "from config.llm_client import test_connection; test_connection()"
    """
    from rich.console import Console
    console = Console()

    console.print(
        f"\n[bold]🔌 Testing OpenRouter connection...[/bold]\n"
        f"  Model: [cyan]{settings.openrouter_model}[/cyan]\n"
        f"  URL:   [dim]{settings.openrouter_base_url}[/dim]\n"
    )

    try:
        response = get_llm_response(
            prompt="Reply with exactly: GITLAB_RAG_OK",
            system_prompt="You are a test assistant. Follow instructions exactly.",
            max_tokens=20,
            temperature=0.0,
        )

        if response and len(response) > 0:
            console.print(
                f"[bold green]✅ OpenRouter connected![/bold green]\n"
                f"  Response: [dim]{response.strip()!r}[/dim]\n"
                f"  Model:    [dim]{settings.openrouter_model}[/dim]\n"
            )
            return True
        else:
            console.print("[red]❌ Got empty response from OpenRouter[/red]")
            return False

    except Exception as e:
        console.print(f"\n[red]❌ OpenRouter connection failed:[/red] {e}")
        console.print(
            "\n[yellow]Checklist:[/yellow]\n"
            "  1. Is OPENROUTER_API_KEY set in your .env?\n"
            "  2. Is the key valid? Check: https://openrouter.ai/keys\n"
            "  3. Is the model name correct? Check: https://openrouter.ai/models\n"
            "  4. Do you have internet access?\n"
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABLE FREE MODELS REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

FREE_MODELS = {
    "nvidia/nemotron-super-49b-v1:free": {
        "description": "NVIDIA's powerful model, great at technical reasoning",
        "context_window": 131072,
        "recommended_for": "Complex CI/CD debugging, multi-step reasoning",
    },
    "meta-llama/llama-3.3-70b-instruct:free": {
        "description": "Meta's flagship open model, very capable",
        "context_window": 131072,
        "recommended_for": "General Q&A, explanations",
    },
    "google/gemma-3-27b-it:free": {
        "description": "Google's Gemma 3 — fast and capable",
        "context_window": 131072,
        "recommended_for": "Fast responses",
    },
    "mistralai/mistral-7b-instruct:free": {
        "description": "Lightweight and fast",
        "context_window": 32768,
        "recommended_for": "Simple queries, fastest responses",
    },
    "deepseek/deepseek-r1:free": {
        "description": "Strong reasoning model from DeepSeek",
        "context_window": 163840,
        "recommended_for": "Complex pipeline debugging, code analysis",
    },
}


def list_free_models() -> None:
    """Print available free models and their capabilities."""
    from rich.table import Table
    from rich.console import Console
    console = Console()

    table = Table(title="🆓 Free Models on OpenRouter", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Best For", style="white")
    table.add_column("Context", style="yellow")

    for model_id, info in FREE_MODELS.items():
        marker = " ← current" if model_id == settings.openrouter_model else ""
        table.add_row(
            model_id + marker,
            info["recommended_for"],
            f"{info['context_window']:,} tokens",
        )

    console.print(table)
    console.print(
        "\n[dim]Change model in .env → OPENROUTER_MODEL=<model_id>[/dim]\n"
    )
