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

Fishery councillor (optional): after each agent proposes a norm, they discuss
how to operationalize it with the fishery councillor — a real `opencode`
agent reached over its HTTP server API. This is off by default (proposals
just skip straight to voting with a blank operationalization). To turn it on:

```
opencode serve --hostname 127.0.0.1 --port 4096   # in another terminal
export OPENCODE_SERVER_URL=http://127.0.0.1:4096
cd backend && uv run uvicorn genfishery.api.app:app --port 8000
```

`OPENCODE_MODEL_ID` defaults to whatever model `LLM_PROVIDER` is already
using for proposals; `OPENCODE_PROVIDER_ID` (default `ollama`) and
`OPENCODE_AGENT` (default `fishery-councillor`) rarely need overriding. See
`opencode.json`/`.opencode/agent/fishery-councillor.md` at the repo root for
its persona and provider config.

Running on Aoraki (Otago's HPC cluster), with Ollama + opencode both
backgrounded in one Slurm job: see [`deploy/README.md`](../deploy/README.md).