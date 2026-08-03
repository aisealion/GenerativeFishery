To run with GPT-5.4 via your LiteLLM proxy (now the default — no code changes needed):


export LITELLM_API_KEY=litellm
# LITELLM_BASE_URL defaults to https://llm.uod.otago.ac.nz/v1 — only set it if different
cd backend && uv run uvicorn genfishery.api.app:app --port 8000
To run with a local Ollama model instead (e.g. gpt-oss:20b):


export LLM_PROVIDER=ollama
# OLLAMA_BASE_URL defaults to http://localhost:11434/v1
cd backend && uv run uvicorn genfishery.api.app:app --port 8000
To fall back to the original Claude path:


export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=<your key>


Port 8000 serves the JSON/WebSocket API directly — there is no bundled UI.


Backend:


export FISHERY_CONFIGS=live_demo_a.yaml
cd backend && uv run uvicorn genfishery.api.app:app --port 8000
(this is also the default when FISHERY_CONFIGS is unset — single fishery, no migration counterpart. Comma-separate for multiple, e.g. FISHERY_CONFIGS=live_demo_a.yaml,live_demo_b.yaml to run two fisheries with migration linking.)

Fishery SE agent (optional): after each agent proposes a norm, they discuss
how to operationalize it with a real `opencode` agent, reached over its HTTP
server API. Whichever proposal wins that round's vote then gets implemented
by a *separate* opencode agent as an actual code change under `backend/`,
tested, and committed with git — two distinct agents/personas
(`.opencode/agent/fishery-discussion-agent.md`,
`.opencode/agent/fishery-code-agent.md`), not one dual-mode agent: the
discussion one has no edit/bash permission at all, so "never touch code
during discussion" is enforced by opencode itself, not just a prompt
instruction. See this file's own module docstring in `sim/engine.py` for the
full round shape. This is off by default (proposals just skip straight to
voting with a blank operationalization, and no code changes ever happen). To
turn it on:

```
opencode serve --hostname 127.0.0.1 --port 4096   # in another terminal
export OPENCODE_SERVER_URL=http://127.0.0.1:4096
cd backend && uv run uvicorn genfishery.api.app:app --port 8000
```

`OPENCODE_MODEL_ID` defaults to whatever model `LLM_PROVIDER` is already
using for proposals; `OPENCODE_PROVIDER_ID` (default `ollama`),
`OPENCODE_DISCUSSION_AGENT` (default `fishery-discussion-agent`), and
`OPENCODE_CODE_AGENT` (default `fishery-code-agent`) rarely need overriding.
See `opencode.json`/`.opencode/agent/fishery-discussion-agent.md`/
`.opencode/agent/fishery-code-agent.md` at the repo root for their
personas and provider config.

Running on Aoraki (Otago's HPC cluster), with Ollama + opencode both
backgrounded in one Slurm job: see [`deploy/README.md`](../deploy/README.md).