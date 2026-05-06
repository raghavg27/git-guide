"""
config/llm_client.py
──────────────────────
LLM client for all CrewAI agents — powered by OpenRouter (free).

TWO CLIENTS, TWO PURPOSES:
  1. get_crewai_llm()   → Returns a crewai.LLM object
                          CrewAI's Agent(llm=...) expects this EXACT type.
                          CrewAI uses LiteLLM internally — NOT LangChain.
                          Model prefix format: "openrouter/model-name"

  2. get_llm_response() → Simple one-shot call, returns plain string.
                          Used for quick calls outside agents (in tools).
                          Uses raw openai SDK directly.

WHY crewai.LLM AND NOT ChatOpenAI?
  CrewAI's Agent pydantic model validates llm= as either:
    - a plain string  (model name)
    - crewai.BaseLLM  (crewai's own LLM wrapper)

  ChatOpenAI from langchain_openai is NEITHER — it fails validation.
  crewai.LLM is the correct type. It wraps LiteLLM under the hood.
  LiteLLM routes to OpenRouter via the "openrouter/<model>" prefix.
"""

from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT 1: crewai.LLM — the ONLY type Agent(llm=...) accepts
# ─────────────────────────────────────────────────────────────────────────────

def get_crewai_llm():
    """
    Returns a crewai.LLM object pointed at OpenRouter.

    crewai.LLM wraps LiteLLM. LiteLLM routes to OpenRouter
    when the model name is prefixed with "openrouter/".

    Usage in agents:
        from config.llm_client import get_crewai_llm
        agent = Agent(llm=get_crewai_llm(), ...)
    """
    from crewai import LLM

    # LiteLLM requires "openrouter/" prefix to route correctly
    model = settings.openrouter_model
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    return LLM(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        extra_headers={
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        },
    )


# Backward-compatible alias (for any file still importing get_llm_client)
get_llm_client = get_crewai_llm


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT 2: Raw OpenAI SDK — for quick one-shot calls inside tools
# ─────────────────────────────────────────────────────────────────────────────

def get_raw_client():
    """
    Returns a raw openai.OpenAI client pointed at OpenRouter.
    Use for one-shot calls OUTSIDE CrewAI agents (e.g., in retrieval tools).
    """
    from openai import OpenAI
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        default_headers={
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        },
    )


def get_async_client():
    """
    Returns an async OpenAI client pointed at OpenRouter.
    Use with asyncio.gather() to fire multiple LLM calls simultaneously.
    """
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        default_headers={
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        },
    )


async def async_get_llm_response(
    prompt: str,
    system_prompt: str = "You are a helpful GitLab documentation expert.",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Async version of get_llm_response.
    Awaitable — use with asyncio.gather() to run multiple calls in parallel.
    """
    client = get_async_client()
    response = await client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )
    text = response.choices[0].message.content or ""
    logger.debug(f"async LLM: {len(text)} chars | model={settings.openrouter_model}")
    return text


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
    Send a prompt to OpenRouter, return plain string response.
    Retried up to 3x on failure.

    Use this for quick LLM calls inside tools (e.g. generating search phrasings).
    Do NOT use inside agent definitions — use get_crewai_llm() there.
    """
    client = get_raw_client()

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
    logger.debug(f"LLM: {len(text)} chars | model={settings.openrouter_model}")
    return text


def get_llm_response_with_history(
    messages: list[dict],
    system_prompt: str = "You are a helpful GitLab documentation expert.",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Send full conversation history to the LLM.
    Used for multi-turn conversations in Phase 3 web UI.
    """
    client = get_raw_client()
    response = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )
    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_connection() -> bool:
    """
    Verify OpenRouter is reachable and both clients work.
    Run: python -c "from config.llm_client import test_connection; test_connection()"
    """
    from rich.console import Console
    console = Console()

    console.print(
        f"\n[bold]🔌 Testing OpenRouter connection...[/bold]\n"
        f"  Model: [cyan]{settings.openrouter_model}[/cyan]\n"
        f"  URL:   [dim]{settings.openrouter_base_url}[/dim]\n"
    )

    # Test 1: Raw client (used in tools)
    try:
        response = get_llm_response(
            prompt="Reply with exactly: OK",
            system_prompt="Reply with exactly what is asked.",
            max_tokens=10,
            temperature=0.0,
        )
        console.print(
            f"  [green]✅ Raw client:[/green] Connected "
            f"| Response: [dim]{response.strip()!r}[/dim]"
        )
    except Exception as e:
        console.print(f"  [red]❌ Raw client failed:[/red] {e}")
        _print_troubleshooting(console)
        return False

    # Test 2: crewai.LLM (used by all agents)
    try:
        llm = get_crewai_llm()
        result = llm.call("Reply with exactly: OK")
        console.print(
            f"  [green]✅ CrewAI LLM:[/green] Connected "
            f"| Response: [dim]{str(result).strip()!r}[/dim]"
        )
    except Exception as e:
        console.print(f"  [red]❌ CrewAI LLM failed:[/red] {e}")
        console.print("  [dim]Ensure crewai is installed: pip install crewai>=0.60.0[/dim]")
        return False

    console.print(
        f"\n[bold green]✅ OpenRouter fully connected![/bold green]\n"
        f"  Model: [cyan]{settings.openrouter_model}[/cyan]\n"
    )
    return True


def _print_troubleshooting(console) -> None:
    console.print(
        "\n[yellow]Troubleshooting:[/yellow]\n"
        "  1. OPENROUTER_API_KEY set in .env?\n"
        "  2. Key valid? → https://openrouter.ai/keys\n"
        "  3. Model name correct? → https://openrouter.ai/models?q=free\n"
        "  4. Internet available?\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FREE MODELS REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

FREE_MODELS = {
    "nvidia/nemotron-super-49b-v1:free": "Powerful, great at technical reasoning (recommended)",
    "meta-llama/llama-3.3-70b-instruct:free": "Meta's flagship, very capable",
    "google/gemma-3-27b-it:free": "Google Gemma, fast",
    "mistralai/mistral-7b-instruct:free": "Lightweight, fastest responses",
    "deepseek/deepseek-r1:free": "Strong reasoning, good for debugging",
}


def list_free_models() -> None:
    """Print available free models. Change via OPENROUTER_MODEL in .env"""
    from rich.table import Table
    from rich.console import Console
    console = Console()

    table = Table(title="🆓 Free Models on OpenRouter", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for model_id, desc in FREE_MODELS.items():
        marker = " ← active" if model_id == settings.openrouter_model else ""
        table.add_row(model_id + marker, desc)

    console.print(table)
    console.print("[dim]Change: OPENROUTER_MODEL=<model_id> in .env[/dim]\n")
    
def get_llm_client():
    """Alias: return the LangChain LLM object used by CrewAI agents"""
    return get_crewai_llm()