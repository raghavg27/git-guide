# Phase 2 Quick Start Guide

## What Was Built

```
🏗️  5 SPECIALIZED AGENTS
│
├─ [1] Intent Classifier      "What is the user really asking?"
├─ [2] Query Rewriter         "How should we search for this?"
├─ [3] Retriever              "What docs match the query?"
├─ [4] Validator              "Are these chunks actually good?"
└─ [5] Synthesiser            "Write the final answer with citations"

🔧 4 RETRIEVAL TOOLS
├─ semantic_search()          Broad semantic similarity search
├─ filtered_search()          Search with metadata filters
├─ multi_query_search()       Multiple phrasings + RRF fusion
└─ get_chunk_by_id()          Fetch specific chunk by ID

💾 KNOWLEDGE BASE (from Phase 1)
└─ ChromaDB with 30,000+ embedded GitLab doc chunks
   (all free, all local, all offline after first run)
```

---

## Try It Now

### 1. Test Your Setup
```bash
# Check OpenRouter connection
python -c "from config.llm_client import test_connection; test_connection()"
```

### 2. Run Interactive Chat
```bash
# Start asking questions!
python -m phase2_agents.run_agents
```

### 3. Test a Single Query
```bash
python -m phase2_agents.run_agents --test-query "how do I cache npm dependencies in gitlab ci?"
```

### 4. Use Simplified Mode (Faster)
```bash
# Skip Intent Classifier + Query Rewriter
# Just Retriever + Synthesiser
python -m phase2_agents.run_agents --simple
```

---

## What to Expect

### Sample Query
```
You: How do I fix exit code 137 in my GitLab CI pipeline?
```

### Behind the Scenes
```
[1] Intent Classifier
    → "This is a DEBUGGING question"

[2] Query Rewriter
    → "gitlab ci exit code 137 out of memory"
    → "why does runner exit with 137"
    → "docker container killed memory limit"

[3] Retriever
    → Searches with all 3 queries
    → Gets 12 chunks from ci/troubleshooting

[4] Validator
    → Keeps 2 highly relevant
    → Warns on 1 supplementary
    → Skips 9 irrelevant

[5] Synthesiser
    → Writes answer: "Exit code 137 means your container was killed..."
    → Adds solutions from validated chunks
    → Cites source URLs
    → Includes version notes
```

### Answer You Get
```
Exit code 137 means your Docker container ran out of memory.

Solution:
  Increase your runner's memory limit to at least 2GB...

Configuration:
  [code example from official docs]

Related: Docker executor, performance tuning
```

---

## Performance

| Mode | Speed | Quality | Cost |
|---|---|---|---|
| **Full** (all 5 agents) | 30-120s | Highest | $0.00 |
| **Simplified** (2 agents) | 10-30s | Good | $0.00 |
| **Direct retrieval** (0 agents) | <5s | Chunks only | $0.00 |

---

## File Structure

```
phase2_agents/
├── tools/
│   └── retrieval_tools.py       ← 4 search functions
├── agents/
│   ├── intent_classifier.py     ← Agent 1
│   ├── query_rewriter.py        ← Agent 2
│   ├── retriever.py             ← Agent 3 (uses tools)
│   ├── validator.py             ← Agent 4 (uses tools)
│   └── synthesiser.py           ← Agent 5
├── crew.py                      ← Orchestrates all agents
└── run_agents.py                ← Main entry point
```

---

## Example Queries to Try

```
✅ "How do I cache npm dependencies in GitLab CI?"
✅ "What's the difference between cache and artifacts?"
✅ "How do I configure a Docker executor for GitLab Runner?"
✅ "How do I migrate from Jenkins to GitLab CI?"
✅ "What permissions does a Developer role have?"
✅ "Why is my pipeline timing out?"
✅ "How do I use the GitLab REST API to list merge requests?"
```

---

## FAQ

**Q: Why 5 agents instead of 1?**
A: Each agent specializes in one task. This improves quality:
   - Classification catches user intent
   - Rewriting optimizes search
   - Retrieval gets diverse results
   - Validation filters noise
   - Synthesis grounds in sources

**Q: How is this different from just ChatGPT?**
A: This system uses YOUR docs as ground truth. Every answer is based on
   official GitLab documentation with citations. No hallucinations.

**Q: Can I use a different LLM?**
A: Yes! Change OPENROUTER_MODEL in .env to any free model on OpenRouter.
   Suggestions: nvidia/nemotron-super-49b-v1:free, meta-llama/llama-3.3-70b

**Q: What if I want just retrieval without agents?**
A: Use the tools directly:
   ```python
   from phase2_agents.tools.retrieval_tools import semantic_search
   results = semantic_search("your query")
   ```

**Q: How long does Phase 2 take to run?**
A: First run: downloads CrewAI + dependencies (~2 min)
   Subsequent runs: 30-120 seconds per query (depending on agent reasoning)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No results found" | Try rephrasing or use --simple mode |
| "Agent thinking forever" | Ctrl+C and try shorter query or --simple |
| "OpenRouter error" | Check API key, check rate limits |
| "Import errors" | `pip install -r requirements.txt` |

---

## Next Steps

- Try the chatbot with real GitLab CI questions
- Adjust agents' system prompts (in agents/*.py files)
- Add new agents (pipeline debugger, admin guide bot, etc.)
- Move to Phase 3: Web UI with Streamlit
