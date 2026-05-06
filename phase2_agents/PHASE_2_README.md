# Phase 2 — CrewAI Agents

**Build a multi-agent system that reasons about your questions and retrieves the best documentation.**

---

## 🏗️ Architecture Overview

### The 5-Agent Pipeline

```
User Question
    ↓
[1] INTENT CLASSIFIER  (Reasoning)
    ↓ Understands what user is asking for
    ↓
[2] QUERY REWRITER     (Reasoning)
    ↓ Generates 3 optimized search queries
    ↓
[3] RETRIEVER          (Search Tool)
    ↓ Searches vector store with multiple strategies
    ↓
[4] VALIDATOR          (Reasoning + Search)
    ↓ Checks quality, relevance, currency of chunks
    ↓
[5] SYNTHESISER        (Writing)
    ↓ Writes final answer grounded in sources
    ↓
Final Answer (with citations)
```

### Each Agent's Job

| Agent | Role | Tools | Input | Output |
|---|---|---|---|---|
| **Intent Classifier** | Understands the question | None (pure reasoning) | Raw user question | Intent category + key concepts + recommended section |
| **Query Rewriter** | Optimizes search | None (pure reasoning) | User question | Primary query + 2-3 alternatives + filter recommendations |
| **Retriever** | Searches knowledge base | `semantic_search`, `filtered_search`, `multi_query_search`, `get_chunk_by_id` | Search query + filters | Top 5-10 relevant chunks from vector store |
| **Validator** | Quality checks chunks | `get_chunk_by_id` (to fetch more context) | Raw retrieved chunks | Validated chunks (✅ keep, ⚠️ warn, ❌ skip) |
| **Synthesiser** | Writes the answer | None (pure writing) | Validated chunks + original question | Final answer with full citations |

---

## 🚀 Quick Start

### Prerequisites
- Phase 1 complete (vector store built)
- OpenRouter API key in `.env`
- `pip install -r requirements.txt` (done in Phase 1)

### Run the Agents

**Interactive mode** — ask questions:
```bash
python -m phase2_agents.run_agents
```

**Single query** — test one question:
```bash
python -m phase2_agents.run_agents --test-query "how do I cache in gitlab ci?"
```

**Simplified mode** — faster, less reasoning:
```bash
python -m phase2_agents.run_agents --simple
```

**Verbose mode** — see all agent thinking:
```bash
python -m phase2_agents.run_agents --verbose
```

---

## 📋 Example Interaction

### User Question
```
"How do I fix exit code 137 in my GitLab CI pipeline?"
```

### What Happens Behind the Scenes

```
[1] INTENT CLASSIFIER
    Intent: DEBUGGING
    Concepts: exit code, pipeline, failure, runner
    Section: ci
    
    → "This is a debugging question about pipeline failure"

[2] QUERY REWRITER
    Primary:    "gitlab ci exit code 137 memory out of memory"
    Alt 1:      "why does gitlab runner exit with code 137"
    Alt 2:      "gitlab ci job killed memory limit exceeded"
    
    → "The 137 exit code usually means out of memory"

[3] RETRIEVER (searches with all 3 queries)
    ✅ "Troubleshooting exit codes" (98% relevance)
    ✅ "Docker executor memory limits" (87% relevance)
    ✅ "CI/CD performance optimization" (72% relevance)
    
    → Retrieved 12 chunks from ci/troubleshooting, ci/docker

[4] VALIDATOR (checks quality)
    ✅ Keep: Chunk 1 (directly answers, current)
    ✅ Keep: Chunk 2 (provides solution, has code example)
    ⚠️  Warn: Chunk 3 (relates but talks about optimization, not debugging)
    
    → "2 highly relevant + 1 supplementary chunk"

[5] SYNTHESISER (writes answer)
    Exit code 137 means the Docker container was killed,
    usually due to out-of-memory. See [link to docs].
    
    To fix:
    1. Increase runner memory: [config from docs]
    2. Optimize Docker build: [best practices from docs]
    3. Use smaller base image: [example from docs]
    
    Version note: Docker limits available since GitLab 13.0
    
    → Final answer with 3 citations
```

### Final Answer (to user)
```
Exit code 137 means your Docker container ran out of memory.

Solution:
  Increase your GitLab Runner's memory limit to at least 2GB...
  
Example configuration:
  [gitlab-runner config from official docs]
  
Related: Docker executor docs, performance tuning
```

---

## 🛠️ Agent Details

### 1. Intent Classifier
- **No tools** — pure reasoning
- **Input:** Raw user question
- **Output:** Intent category + concepts
- **Categories:** DEBUGGING, HOW_TO, EXPLANATION, TROUBLESHOOT, API, MIGRATION, PERMISSIONS, VERSION_INFO
- **Why:** Helps downstream agents understand context

### 2. Query Rewriter
- **No tools** — pure reasoning
- **Input:** User question + intent
- **Output:** Primary query + 2-3 alternatives + filter suggestions
- **Technique:** Generates multiple phrasings to improve recall
- **Why:** Different people ask the same thing different ways

### 3. Retriever
- **Tools:** 4 search strategies
  - `semantic_search()` — broad semantic similarity
  - `filtered_search()` — with metadata filters (section, has_code, etc.)
  - `multi_query_search()` — generates alternatives + fuses with RRF
  - `get_chunk_by_id()` — fetch specific chunk
- **Input:** Search queries + recommended filters
- **Output:** Top 5-15 retrieved chunks
- **Decision Logic:**
  ```
  if query_rewriter_recommended_section:
    use filtered_search(section=...)
  elif question_complex:
    use multi_query_search()
  else:
    use semantic_search()
  ```
- **Why:** Multiple search strategies catch different relevant docs

### 4. Validator
- **Tools:** `get_chunk_by_id()` (to fetch more context)
- **Input:** Retrieved chunks + original question
- **Output:** Quality assessment
  - ✅ **KEEP** — Highly relevant, current, complete
  - ⚠️ **WARN** — Relevant but has caveats (deprecated, unclear)
  - ❌ **SKIP** — Not relevant, off-topic
  - 🔗 **EXPAND** — Need related chunks
- **Checks:**
  1. **Relevance** — Does it answer the question?
  2. **Freshness** — Is it current? (check deprecated flag, version)
  3. **Completeness** — Enough info? Or fragmented?
  4. **Accuracy** — Correct and trustworthy?
  5. **Coverage** — Related chunks we should include?
- **Why:** Filters out low-quality, off-topic, outdated chunks

### 5. Synthesiser
- **No tools** — pure writing
- **Input:** Validated chunks + original question
- **Output:** Final answer with full citations
- **Structure:**
  1. **Quick answer** (1 sentence)
  2. **Explanation** (2-4 paragraphs with sources)
  3. **Example/Code** (if applicable, from chunks)
  4. **Caveats & warnings** (version, deprecation, edge cases)
  5. **Related resources** (links from chunk metadata)
- **Citation Rules:**
  - Every technical fact has a source
  - Include URL, not just "GitLab docs"
  - Call out deprecated features explicitly
  - Never extrapolate beyond what chunks say
  - Never hallucinate
- **Why:** Grounds answer in sources, provides citations

---

## 🎯 Advanced Usage

### Use Different Crews

**Full crew** (all 5 agents):
```python
from phase2_agents.crew import create_gitlab_rag_crew
crew = create_gitlab_rag_crew()
result = crew.kickoff(inputs={"user_query": "your question"})
```

**Simplified crew** (just Retriever + Synthesiser):
```python
from phase2_agents.crew import create_simple_retrieval_crew
crew = create_simple_retrieval_crew()  # Faster, less reasoning
result = crew.kickoff(inputs={"user_query": "your question"})
```

### Programmatic Usage

```python
from phase2_agents.crew import create_gitlab_rag_crew

crew = create_gitlab_rag_crew()

questions = [
    "how do I cache npm dependencies?",
    "what is the difference between cache and artifacts?",
    "how do I configure a docker runner?",
]

for q in questions:
    result = crew.kickoff(inputs={"user_query": q})
    print(result)
    print("\n" + "="*80 + "\n")
```

### Custom Tool Usage

If you want to use just the retrieval tools directly (without agents):

```python
from phase2_agents.tools.retrieval_tools import semantic_search, filtered_search

# Simple search
results = semantic_search("how do I cache dependencies?")
print(results)

# Filtered search (CI docs only, with code examples)
results = filtered_search(
    "caching strategy",
    section="ci",
    has_code=True,
    n_results=5
)
print(results)
```

---

## 📊 What to Expect

### Performance
- **Full crew:** 30-120 seconds per query (lots of reasoning)
- **Simplified crew:** 10-30 seconds per query (fast)
- **Just retrieval:** <5 seconds (retrieval only, no synthesis)

### Quality
- **Hallucinations:** Near zero (all claims grounded in chunks)
- **Relevance:** 85-95% (if Phase 1 chunking is good)
- **Citations:** 100% (every fact has a source)

### Cost
- **Full cost:** $0.00 (OpenRouter free tier + local embeddings)
- **Rate limits:** Generous on free tier (no charge)

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| "No module named crewai" | `pip install -r requirements.txt` |
| "Vector store not found" | Run Phase 1: `python -m phase1_ingestion.run_agents` |
| "OpenRouter connection failed" | Check OPENROUTER_API_KEY in .env |
| "Agent seems stuck / timeout" | CrewAI may be thinking. Increase timeout or use --simple |
| "Empty / irrelevant answers" | Phase 1 chunks may be incomplete. Check chunk quality |

---

## 🔜 Next Phase

Phase 3 (coming next):
- **Hybrid retrieval** — combine semantic + BM25 keyword search
- **Re-ranking** — order results by true relevance
- **Streaming** — stream answers as they're generated

Phase 4 (coming next):
- **Web UI** — Streamlit interface
- **Chat history** — multi-turn conversations
- **Feedback loops** — improve answers based on user ratings

---

## 📚 Files in Phase 2

```
phase2_agents/
├── tools/
│   ├── __init__.py
│   └── retrieval_tools.py          # 4 search tools
├── agents/
│   ├── __init__.py
│   ├── intent_classifier.py        # Agent 1
│   ├── query_rewriter.py           # Agent 2
│   ├── retriever.py                # Agent 3
│   ├── validator.py                # Agent 4
│   └── synthesiser.py              # Agent 5
├── __init__.py
├── crew.py                         # Orchestrates all agents
└── run_agents.py                   # Main entry point
```

---

## 🎓 Understanding the Flow

**Why this specific order?**

1. **Intent first** — Tells everyone what we're solving
2. **Query rewrite** — Ensures retrieval is optimized
3. **Retrieve** — Gets raw material (might be noisy)
4. **Validate** — Filters to high-quality only
5. **Synthesise** — Builds answer from vetted sources

**Why not skip steps?**

- Skip Intent → agents don't understand context
- Skip Query Rewrite → poor search results (user phrasing is imperfect)
- Skip Validate → hallucinations (bad chunks in final answer)
- Skip Synthesise → raw chunks (unfriendly, unsummarised)

Each step removes risk and improves quality.

---

## 💡 Tips for Best Results

1. **Ask clear questions**
   - ❌ "this doesnt work"
   - ✅ "exit code 137 in gitlab ci pipeline"

2. **Mention context**
   - ❌ "how to configure something"
   - ✅ "how to configure docker executor for gitlab runner"

3. **Include error messages**
   - ❌ "my build failed"
   - ✅ "my gitlab ci build failed with yaml syntax error: unknown keys"

4. **Use simplified crew for quick answers**
   - Full crew = thorough but slow
   - Simplified = fast, good for simple questions
