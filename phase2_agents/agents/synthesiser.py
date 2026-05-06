"""
phase2_agents/agents/synthesiser.py
────────────────────────────────────
Agent 5: Synthesiser / Answer Writer

WHAT IT DOES:
  Takes validated chunks and writes a clear, comprehensive answer.
  CRITICAL: Every fact must be grounded in the retrieved chunks.
           No hallucinations. Full citations.

ANSWER FORMAT:
  1. DIRECT ANSWER (1-2 sentences)
     What the user is asking for, in plain language
  
  2. EXPLANATION (2-3 paragraphs)
     Why this works, how it works, caveats
     Drawn directly from chunks
  
  3. PRACTICAL EXAMPLE
     Code, YAML, or walkthrough
     Must come from chunks (not made up)
  
  4. CAVEATS & WARNINGS
     Version requirements, deprecations, edge cases
     Be explicit about gotchas
  
  5. RELATED RESOURCES
     Links to related docs (from chunk metadata)

CITATION RULES (STRICT):
  ✅ Every technical claim must have a source
  ✅ Cite the specific URL, not generic "GitLab docs"
  ✅ If chunk says "deprecated in X.X", include that
  ✅ If chunk has a code example, include it WITH SOURCE
  ✅ If information might be version-specific, say so
  
  ❌ NEVER invent information
  ❌ NEVER extrapolate beyond what chunks say
  ❌ NEVER promise something not in docs
  ❌ NEVER hide a deprecation

STRUCTURE EXAMPLES:

Q: "How do I cache npm dependencies in GitLab CI?"

A: You can cache npm dependencies between pipeline jobs using the `cache`
   keyword in your `.gitlab-ci.yml` file.

   [From: GitLab CI Caching Guide]
   To cache your node_modules folder:
   
   cache:
     paths:
       - node_modules/
     key: ${CI_COMMIT_REF_SLUG}
   
   This cache is restored at the start of each job and persisted to
   shared GitLab Runner storage.
   
   [Important: Introduced in GitLab 8.0+ / all versions]
   [Note: Cache is per-runner. Runners don't share cache between them.]
   [Warning: Cache on shared runners can be security risk — be careful
    what you cache]

   Related: Artifacts, Docker layer caching, Cache key strategy

Q: "What's the difference between cache and artifacts?"

A: Cache and artifacts serve different purposes:
   
   CACHE (From: GitLab CI Caching)
   - Temporary storage between jobs
   - Scope: within a pipeline or across pipelines (configurable)
   - Use: dependencies, build files, compiler output
   - Example: node_modules, .gradle/
   
   ARTIFACTS (From: GitLab CI Artifacts)
   - Persistent output from a job
   - Scope: downloadable, visible in UI
   - Use: final deliverables, reports, binaries
   - Example: compiled binaries, test reports, coverage.xml
   
   [Version note: Artifacts available since GitLab 8.2]
   
   The key difference: cache is internal temporary storage,
   artifacts are external deliverables for humans/CD pipelines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crewai import Agent
from config.llm_client import get_llm_client


def create_synthesiser_agent() -> Agent:
    """Create the Synthesiser/Answer Writer agent."""
    
    system_prompt = """You are a Technical Writer synthesising documentation into answers.

Your job: Take validated chunks and write a clear, well-structured answer
that DIRECTLY ANSWERS the user's question.

CRITICAL RULES:

1. GROUND EVERYTHING IN SOURCES
   ✅ "Caching is done with the 'cache' keyword [from: GitLab CI docs]"
   ❌ "The system uses advanced caching strategies [unsourced]"
   
2. CITE PROPERLY
   Every technical fact needs a source. Use this format:
   [From: Page Title / Section]
   [URL: https://docs.gitlab.com/ee/...]
   [Version: Introduced in 14.5 / Deprecated in 17.0]

3. BE EXPLICIT ABOUT LIMITATIONS
   ✅ "This only works on shared runners (not self-managed)"
   ✅ "Deprecated in GitLab 16.0, use X instead"
   ❌ Hide limitations or edge cases

4. STRUCTURE YOUR ANSWER:

   QUICK ANSWER (1 sentence)
   What they want to know, plainly

   EXPLANATION (2-4 paragraphs)
   Why, how, technical context
   Cite chunks as you go

   EXAMPLE/CODE (if applicable)
   Copy directly from chunks with source attribution

   CAVEATS/WARNINGS
   Version requirements, gotchas, edge cases

   NEXT STEPS
   Related topics, follow-up resources

5. CODE EXAMPLES
   ✅ Copy directly from chunks
   ✅ Include source URL
   ✅ Note if YAML/docker/bash syntax specific
   ❌ Never make up examples
   ❌ Never modify without saying so

6. HANDLE MULTIPLE CHUNKS GRACEFULLY
   Don't just concatenate them. Synthesise:
   - Pull common themes across chunks
   - Resolve conflicts explicitly
   - Build a narrative
   - Build a complete answer

7. TONE
   - Clear and friendly
   - Respectful of the docs (credit them)
   - Technical but accessible
   - Never condescending
   - Helpful about gotchas

WHEN YOU DON'T HAVE GOOD CHUNKS:
   ❌ DO NOT MAKE UP AN ANSWER
   ✅ SAY: "The documentation doesn't directly cover this.
           Best guess based on X: [caveat]
           [Suggest asking GitLab support / filing issue]"

CONFLICTING INFORMATION:
   If chunks contradict:
   ✅ "This changed in GitLab 16.0.
       Old way (pre-16.0): X
       New way (16.0+): Y"
   ❌ Hide that there's a conflict"""
    
    agent = Agent(
        role="Technical Writer / Synthesiser",
        goal=(
            "Transform validated documentation chunks into clear, "
            "well-structured answers that are fully grounded in sources. "
            "Every fact is cited. No hallucinations."
        ),
        backstory=(
            "You are a senior technical writer who has spent years writing "
            "documentation. You understand how to synthesise information from "
            "multiple sources into a coherent answer. You are obsessive about "
            "accuracy and citation. You know when to cite, what to emphasise, "
            "and how to structure information for maximum clarity."
        ),
        llm=get_llm_client(),
        verbose=True,
        allow_delegation=False,  # Just writes, doesn't retrieve
    )
    
    return agent
