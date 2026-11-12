# Meta Agent

Meta Agent compiles a YAML agent-network definition into a LangGraph workflow. The compiler supports typed state, routing, human approval nodes, swarm and consumer execution, context policies, retries, tool safeguards, evaluators, and guarded harness optimization.

## Requirements

- Python 3.11 or newer
- A Gemini API key for the default example model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` before running the live compiler. The example network also defines optional `JOB_API_KEY` and `APPLICATION_API_KEY` credentials.

## Run

The example network definition is [tests/jobsearch.agent.yaml](tests/jobsearch.agent.yaml). Run the compiler from the repository root:

```bash
python compiler.py
```

The compiler prints the generated graph and pauses when a human-in-the-loop node requires input.

## Test

Tests use the standard library `unittest` runner and do not require model credentials:

```bash
python -m unittest discover -s tests -q
```

## Layout

- Root Python modules: compiler, graph construction, state, memory, tools, resilience, evaluators, and optimization.
- `tests/`: YAML example network, model/tool fixtures, and unit tests.
