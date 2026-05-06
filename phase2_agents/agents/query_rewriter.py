"""
phase2_agents/agents/query_rewriter.py
───────────────────────────────────────
Agent 2: Query Rewriter

WHAT IT DOES:
  Takes a vague/casual user question and rewrites it into multiple
  PRECISE, SEARCH-OPTIMIZED versions.

WHY REWRITING MATTERS:
  User says: "my pipeline is slow"
  Rewritten versions:
    1. "GitLab CI pipeline performance optimization"
    2. "how to speed up slow gitlab-ci builds"
    3. "pipeline timeout configuration timeout settings"
  
  All 3 retrieve different docs → better coverage

EXAMPLES OF REWRITES:
  User: "how do I cache stuff in ci?"
  Rewritten:
    1. "gitlab ci caching configuration artifact cache"
    2. "cache dependencies between pipeline jobs"
    3. "docker layer caching in gitlab ci"

ALSO IDENTIFIES:
  - Metadata filters to apply (section, version, etc.)
  - Whether this should be a simple or multi-query search
  - Any domain-specific terms to emphasise (runner, yaml, etc.)

OUTPUT:
  Primary Query:
    "GitLab CI caching configuration"
  
  Alternative Phrasings:
    1. "how to cache dependencies gitlab runner"
    2. "artifact caching between pipeline jobs"
    3. "docker build cache in gitlab ci"
  
  Filter Recommendations:
    Section: ci
    Has Code: true (helpful to see examples)
    Exclude Deprecated: yes
  
  Confidence: HIGH
  Rationale: User is clearly asking about CI/CD caching...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crewai import Agent
from config.llm_client import get_llm_client


def create_query_rewriter_agent() -> Agent:
    """Create the Query Rewriter agent."""
    
    system_prompt = """You are a Query Rewriter for a documentation search system.

Your job is to take a user's casual/vague question and rewrite it into
MULTIPLE precise, search-optimized versions.

KEY PRINCIPLES:
  1. EXPLICITNESS: Add missing context
     User: "how do I cache stuff?"
     Better: "how to cache dependencies in gitlab ci yaml"
  
  2. DOMAIN TERMS: Use GitLab-specific terminology
     User: "speed up my builds"
     Better: "gitlab ci pipeline performance optimization"
  
  3. MULTIPLE ANGLES: Generate 2-3 phrasings that catch different docs
     Original: "runner not connecting"
     Alt 1: "gitlab runner registration connection issues"
     Alt 2: "runner authentication token configuration"
     Alt 3: "docker executor runner connection troubleshooting"
  
  4. REMOVE NOISE: Strip emotion, remove filler
     User: "ugh why is my pipeline always timing out"
     Better: "gitlab ci pipeline timeout configuration limits"

ALSO RECOMMEND:
  - Which section to search (ci, api, admin, etc.)
  - Whether to require code examples (has_code=true)
  - Confidence level in your rewrite (HIGH/MEDIUM/LOW)

OUTPUT FORMAT:
  Primary Search Query:
  <most important rewrite, optimised for search>
  
  Alternative Phrasings (2-3 options):
    1. <phrasing 1>
    2. <phrasing 2>
    3. <phrasing 3>
  
  Filter Recommendations:
  Section: <ci|api|admin|runner|user|security|topics|update|none>
  Code Examples: <YES|NO|OPTIONAL>
  
  Confidence: <HIGH|MEDIUM|LOW>
  
  Rationale:
  <explain your reasoning>"""
    
    agent = Agent(
        role="Query Rewriter",
        goal=(
            "Transform user questions into multiple search-optimised queries "
            "that will retrieve the most relevant documentation."
        ),
        backstory=(
            "You are an expert search optimizer and GitLab domain specialist. "
            "You understand how people ask questions informally, and you know "
            "how to translate them into precise technical queries. "
            "You generate multiple search angles to catch all relevant docs."
        ),
        llm=get_llm_client(),
        verbose=True,
        allow_delegation=False,  # Pure reasoning, no tools
    )
    
    return agent
