"""
phase2_agents/agents/intent_classifier.py
───────────────────────────────────────────
Agent 1: Intent Classifier

WHAT IT DOES:
  Reads the user's question and determines WHAT they're asking for.
  
INTENT CATEGORIES:
  - DEBUGGING      "why is my pipeline failing?" / "exit code 137"
  - HOW_TO         "how do I configure X?" / "how to cache Y?"
  - EXPLANATION    "what is X?" / "explain GitLab CI"
  - TROUBLESHOOT   "my runner won't connect" / "permission denied"
  - API            "list merge requests using API" / "REST endpoint"
  - MIGRATION      "migrate from Jenkins" / "GitHub Actions to GitLab"
  - PERMISSIONS    "can a developer push to protected branch?"

WHY A SEPARATE INTENT AGENT?
  Knowing the intent helps downstream agents:
  - Debugger agent activates for DEBUGGING intents
  - Filter to API docs for API intents
  - Filter to admin docs for admin intents
  - Synthesiser chooses format based on intent

OUTPUT:
  "The user is asking a DEBUGGING question about pipeline failure.
   Key concepts: pipeline, exit code, failure, runner
   Likely section: ci/troubleshooting
   Best tool: semantic_search (retrieve error handling docs)"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crewai import Agent
from config.llm_client import get_llm_client
from config.settings import settings


def create_intent_classifier_agent() -> Agent:
    """
    Create the Intent Classifier agent.
    
    This agent does NOT use tools — it's a pure reasoning agent.
    It reads the user question and categorises it.
    """
    
    system_prompt = """You are an Intent Classifier for a GitLab documentation chatbot.

Your job is to understand what the user is REALLY asking for and categorise their intent.

INTENT CATEGORIES:
  1. DEBUGGING       - User reports a failing pipeline / error / exit code
  2. HOW_TO          - User wants to know how to configure/use a feature
  3. EXPLANATION     - User wants to understand a concept or feature
  4. TROUBLESHOOT    - User has a problem (runner connection, permissions, etc.)
  5. API             - User is asking about REST/GraphQL API endpoints
  6. MIGRATION       - User wants to migrate from another tool
  7. PERMISSIONS     - User asking about roles/access control
  8. VERSION_INFO    - User asking about versions, deprecation, changelog
  9. UNKNOWN         - You can't determine the intent

ALSO IDENTIFY:
  - Key technical concepts mentioned (pipeline, runner, cache, merge request, etc.)
  - Likely GitLab section (ci, api, admin, user, runner, etc.)
  - Whether the user has code/YAML to debug
  - Urgency level (normal, high — are they blocked?)

OUTPUT FORMAT:
  Intent: <CATEGORY>
  Confidence: <HIGH|MEDIUM|LOW>
  Key Concepts: <list of technical terms>
  Likely Section: <ci|api|admin|user|runner|security|topics|update>
  Has Code: <YES|NO>
  
  Analysis:
  <your reasoning about why this is the intent>
  
  Next Steps:
  <what the next agent should do>"""
    
    agent = Agent(
        role="Intent Classifier",
        goal=(
            "Accurately understand and categorise the user's intent. "
            "Identify the core question beneath surface-level wording."
        ),
        backstory=(
            "You are an expert at understanding what people are really asking. "
            "You work for a GitLab help desk and can quickly identify whether "
            "someone needs debugging help, a tutorial, API docs, or something else. "
            "You provide clear categorisation that helps downstream agents do their jobs."
        ),
        llm=get_llm_client(),
        verbose=True,
        allow_delegation=False,  # No tools, just reasoning
    )
    
    return agent
