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
(comma-separate for multiple, e.g. FISHERY_CONFIGS=live_demo_a.yaml,live_demo_b.yaml — that's also the default when unset). With only one fishery, there's no migration counterpart, so migration just never triggers — everything else works the same.