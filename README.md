# Meta Agent

[![PyPI](https://img.shields.io/pypi/v/py-meta-agent.svg)](https://pypi.org/project/py-meta-agent/)
[![Python](https://img.shields.io/pypi/pyversions/py-meta-agent.svg)](https://pypi.org/project/py-meta-agent/)

**Define production-grade agent networks in YAML. Compile them to LangGraph.**

Meta Agent turns a declarative `*.agent.yaml` specification into an executable [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`. You describe the state, tools, agents, and graph topology; the compiler handles the orchestration boilerplate: typed state and reducers, parallel swarm fan-out, human-in-the-loop interrupts, safe conditional routing, context management, retries, self-healing, evaluators, and guarded harness optimization.

```bash
pip install py-meta-agent
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Generate a Network from a Description](#generate-a-network-from-a-description)
- [Compile and Run](#compile-and-run)
- [Programmatic Usage](#programmatic-usage)
- [Core Concepts](#core-concepts)
- [YAML Schema Reference](#yaml-schema-reference)
- [Modules and Project Layout](#modules-and-project-layout)
- [Safety and Security Notes](#safety-and-security-notes)
- [Development and Testing](#development-and-testing)
- [Troubleshooting](#troubleshooting)

---

## Features

| Area | What you get |
| :--- | :--- |
| **Declarative design** | One YAML file describes state, tools, agents, evaluators, and the node graph. |
| **Typed state** | Pydantic-backed state fields with built-in (`append`, `dedupe_by_id`) and custom reducers; worker-scoped fields for parallel runs. |
| **Execution patterns** | Standard agent nodes, periodic nodes, parallel **swarm** fan-out (`Send`), sequential **consumers**. |
| **Human in the loop** | Input, review/approval, and agent-assisted decision interrupts, including auto-approval rules. |
| **Safe routing** | Conditions are evaluated by an AST visitor, never `eval()`. |
| **Context policies** | Pre-filters (e.g. strip PII), few-shot retrieval from memory, window management (`summarize_oldest`, `drop_oldest`, `truncate`, `sliding_window`). |
| **Resilience** | Retries with backoff and jitter, per-tool rate limiting, per-node error routing. |
| **Self-healing** | Trace-driven graph patches with versioning and rollback. |
| **Self-improvement** | Metric targets, versioned harness registry, guarded mutation with approval, offline-eval gating, and automatic rollback. |
| **Scaffolding** | A CLI builder generates a valid `.agent.yaml` plus companion Python modules from a natural-language description. |

---

## Installation

**Requirements:** Python `>= 3.11`.

```bash
pip install py-meta-agent
```

Or install from source for development:

```bash
git clone <your-repository-url>
cd meta-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Configure credentials:

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` (or the credentials for whichever LLM provider you configure). Tools that declare `auth.env` (for example `JOB_API_KEY` or `APPLICATION_API_KEY`) read their keys from the environment as well.

**Runtime dependencies:** `langgraph>=1.0`, `langchain>=1.0`, `langchain-core>=1.0`, `langchain-google-genai>=4.0`, `pydantic>=2.0`, `pyyaml>=6.0`, `jinja2>=3.1`.

**Installed commands:**

| Command | Purpose |
| :--- | :--- |
| `meta-agent` | Compile and run an `.agent.yaml` file. |
| `meta-agent-build` | Generate an `.agent.yaml` and companion modules from a description. |

---

## Quickstart

Create `hello.agent.yaml`:

```yaml
schema_version: "1.0"
name: Hello Agent
entry_node: ask

state:
  question: { type: text }
  answer:   { type: text }

agents:
  answerer:
    description: Answers a question concisely.
    input:  { question: state.question }
    output: state.answer
    harness:
      system_instruction: "Answer concisely and accurately."
      llm_config: { provider: google-genai, model: gemini-3.6-flash, temperature: 0.2 }
      input_template: |
        Question: {{ input.question }}

nodes:
  - id: ask
    type: human_in_the_loop
    input_type: text
    input_state: question
    routing:
      - { when: "state.question != null", to: respond }

  - id: respond
    agent: answerer
    routing:
      - { when: "true", to: end }
```

Run it:

```bash
meta-agent hello.agent.yaml
```

The compiler prints the graph, prompts you at each human-in-the-loop interrupt, and prints the final execution trace.

For a complete multi-agent reference (swarm analysis, review gate, periodic search, optimization), see [`tests/jobsearch.agent.yaml`](tests/jobsearch.agent.yaml).

---

## Generate a Network from a Description

`meta-agent-build` calls an LLM to generate and validate an `.agent.yaml`, then generates and syntax-checks the companion modules (`data_models.py`, `tool_defs.py`, and `hooks.py` / `evaluators.py` when referenced) in the same directory.

```bash
meta-agent-build "Build a customer support triage network with intake, specialist swarm analysis, human approval, and reply delivery" \
  --name "Support Triage" \
  --agents "intake_agent,specialist_agent,reply_agent" \
  --features "swarm,human_in_the_loop,self_healing,optimization" \
  -o support/support_triage.agent.yaml
```

Interactive wizard:

```bash
python -m meta_agent_madcoderme.builder -i
```

| Option | Description |
| :--- | :--- |
| `--yaml-only` | Generate only the `.agent.yaml`, skip companion modules. |
| `--provider`, `--model`, `--temperature` | Builder LLM (defaults: `google-genai`, `gemini-3.6-flash`, `0.2`). |
| `-o <path>` | Output file; must end in `.agent.yaml`. |
| `-i` | Interactive wizard. |

Programmatic equivalents: `generate_agent_yaml(spec)` and `generate_companion_modules(yaml_text, spec)` from `meta_agent_madcoderme`.

> Review generated YAML and tool implementations before running them, especially tools with external side effects.

---

## Compile and Run

```bash
meta-agent path/to/network.agent.yaml
# or
python -m meta_agent_madcoderme.compiler path/to/network.agent.yaml
```

The compiler:

1. Validates that the path exists and ends with `.agent.yaml`.
2. Compiles typed state, tools, agents (merged with `harness_defaults`), memory stores, resilience and self-healing wrappers, evaluators, and the `StateGraph`.
3. Prints the ASCII graph.
4. Executes the workflow, prompting on every human-in-the-loop `__interrupt__`.
5. Prints the final execution trace.

---

## Programmatic Usage

```python
import yaml
from langgraph.types import Command

from meta_agent_madcoderme import (
    create_states,
    create_tools,
    create_agents,
    create_memory,
    create_graph,
)

with open("path/to/network.agent.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

modules = cfg.get("modules", {})
runtime_cfg = cfg.get("runtime", {})
memory_cfg = runtime_cfg.get("memory")

# 1. Build typed state, tools, agents, and memory backends
network_state = create_states(cfg.get("state", {}), modules_config=modules)
tools = create_tools(cfg.get("tools", {}), modules_config=modules)
agents = create_agents(
    cfg.get("agents", {}),
    tools,
    harness_defaults=cfg.get("harness_defaults", {}),
)
checkpointer, store = create_memory(memory_cfg)

# 2. Compile the LangGraph StateGraph
workflow = create_graph(
    network_state=network_state,
    agent_list=agents,
    network_nodes=cfg.get("nodes", []),
    memory_config=memory_cfg,
    tools=tools,
    entry_node=cfg.get("entry_node"),
    runtime_config=runtime_cfg,
    evaluator_config=cfg.get("evaluators", {}),
    modules_config=modules,
)
compiled = workflow.compile(checkpointer=checkpointer, store=store)

# 3. Invoke with initial state and a thread config
run_config = {"configurable": {"thread_id": "session-1"}}
initial_state = {
    k: [] if k == "messages" else None
    for k in network_state.__annotations__.keys()
}
initial_state["resume_file"] = "/path/to/resume.txt"

result = compiled.invoke(initial_state, run_config)

# 4. Handle human-in-the-loop interrupts
while "__interrupt__" in result:
    prompt_info = result["__interrupt__"][0].value
    user_reply = input(f"{prompt_info.get('prompt')}: ")
    result = compiled.invoke(Command(resume=user_reply), config=run_config)
```

When `runtime.self_healing.enabled` is `true`, the compiled workflow also exposes `workflow.__graph_registry__` (a `GraphRegistry`) and `workflow.__self_healer__` (a `SelfHealer`).

---

## Core Concepts

- **State**: shared fields (and optional worker-scoped fields) that flow through the graph. Reducers define how concurrent writes merge.
- **Tools**: callables with typed schemas, optional rate limits, auth, and retries. Agents reference them by name.
- **Agents**: an LLM plus a *harness* (system instruction, model config, context policy, input template) with mapped inputs and a single output state key. Every harness inherits `harness_defaults` through a deep merge.
- **Nodes**: graph vertices that run agents or collect human input. Routing between nodes is declarative.
- **Evaluators and optimization**: evaluators emit metrics; each agent's `optimization` block compares metrics to targets and proposes guarded, versioned harness changes.
- **Self-healing**: when a node fails after retries, a healer model reads the execution trace and proposes a minimal graph patch, recorded as a new graph version that can be rolled back.

---

## YAML Schema Reference

Schema version: `"1.0"`. A full example lives in [`tests/jobsearch.agent.yaml`](tests/jobsearch.agent.yaml).

### Top-level keys

| Key | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `schema_version` | string | Yes | Schema version identifier, e.g. `"1.0"`. |
| `name` | string | No | Human-readable network name. |
| `description` | string | No | Description of the network. |
| `entry_node` | string | Yes* | ID of the starting node. (*Or set `is_starting_node: true` on a node.) |
| `modules` | object | No | Maps logical module roles to Python module names. |
| `runtime` | object | No | Global memory, resilience, and self-improvement policies. |
| `harness_defaults` | object | No | Default harness inherited by all agents via deep merge. |
| `state` | object | Yes | Shared and worker-scoped state fields. |
| `tools` | object | No | Tool definitions, bound to agents by name. |
| `evaluators` | object | No | Evaluators referenced by agent `optimization` blocks. |
| `agents` | object | Yes | Agent definitions (models, harnesses, I/O mappings, retries, optimization). |
| `nodes` | list | Yes | Graph nodes, execution types, iterators, and routing. |

### `modules`

Maps logical roles to Python modules. `utils.resolve_module()` searches standard imports, `tests.<name>`, and `<name>.py` in the working directory or `tests/`.

```yaml
modules:
  data_models: data_models   # Pydantic models referenced by state and tool input_schema
  tools: tool_defs           # Callable tool implementations
  evaluators: evaluators     # Evaluator functions
  hooks: hooks               # Optional custom state reducer functions
```

### `runtime`

Global policies for memory, resilience, self-healing, and guarded self-improvement.

```yaml
runtime:
  memory:
    stores:
      short_term:
        type: in_memory        # in_memory | postgres
        connection_string: ""  # required when type: postgres
        scope: run
      long_term:
        type: in_memory
        connection_string: ""
        scope: global
        namespaces: [successful_parses, accepted_suggestions, feedback]

  resilience:
    default_retry:
      max_attempts: 3
      backoff: exponential
      initial_delay_s: 1
      max_delay_s: 30
      jitter: 0.0
      retry_on: [timeout, rate_limit, transient_error, schema_violation]
    default_timeout_s: 60
    on_unhandled_error: route_to_healer

  self_healing:
    enabled: true
    endpoint: "/heal"
    healer_model:
      provider: google-genai
      model: gemini-3.6-pro
      temperature: 0.1
      system_instruction: "Inspect the execution trace and node error, then output a minimal JSON graph patch to recover."
    versioning:
      enabled: true
      max_versions: 10
      auto_rollback_on_failure: true
    strategies: [retry, fallback_harness, fallback_model, modify_graph, escalate_to_human]
    max_heal_attempts_per_node: 2

  self_improving:
    enabled: true
    guardrails:
      require_versioning: true
      auto_rollback_on_regression: true
      regression_tolerance: 0.0
      max_mutation_depth: 3
      human_approval_for: [system_instruction]
      min_samples_before_mutation: 20
      mutable_fields: [system_instruction, input_template, llm_config.temperature, context_policy.few_shot]
      require_offline_eval: false
```

`concurrency` and `monitoring` blocks may also be declared under `runtime` for configuration completeness.

**Self-healing behavior.** On an unhandled node error, `SelfHealer` reads the execution trace (`_trace` and `messages`) and proposes a JSON graph patch (`node_updates`, `add_nodes`, `remove_nodes`, `reroute`, `goto`). Each patch is recorded as a new `GraphVersion` (`0 -> 1 -> ...`), earlier versions are marked `superseded`, and `rollback()` is supported. Execution continues through LangGraph `Command(goto=..., update={"_error": ..., "_heal": ...})`.

### `harness_defaults`

Inherited by every agent. Agent harnesses are merged with `utils.deep_merge(harness_defaults, agent_harness)`.

```yaml
harness_defaults:
  id: default
  llm_config:
    provider: google-genai     # also accepts model_provider
    model: gemini-3.6-flash
    temperature: 0.2
  fallback_llm_config:
    model: gemini-3.5-flash
    temperature: 0.0
  system_instruction: "You are a precise, helpful assistant."
  output_validation:
    enforce_schema: true
    on_invalid: retry
  context_policy:
    window_management:
      max_tokens: 8192
      overflow_strategy: summarize_oldest   # summarize_oldest | drop_oldest | truncate | sliding_window
```

### `state`

Defines fields on the compiled `NetworkState` `TypedDict`. A `messages` field with LangGraph's `add_messages` reducer is included automatically.

```yaml
state:
  resume_file:        { type: file }
  parsed_resume:      { type: ResumeModel }
  job_listings:       { type: "array[JobListingModel]",  reducer: dedupe_by_id }
  job_analysis:       { type: "array[JobAnalysisModel]", reducer: append }
  single_job_listing: { type: JobListingModel, scope: worker }
```

| Property | Values / behavior |
| :--- | :--- |
| `type` | A class name from `modules.data_models`, `array[<Type>]` (mapped to `List[<Type>]`), or any unmapped primitive name (defaults to `Any`). |
| `reducer` | `append`, `dedupe_by_id` (by `id` key or `.id` attribute), or the name of a custom function in `modules.hooks`. |
| `scope` | `global` (default) or `worker`. Recorded on `NetworkState.__state_scopes__`. |

### `tools`

```yaml
tools:
  job_listings_fetcher:
    type: api_caller
    impl: tools.job_listings_fetcher
    description: Fetches job listings matching preferences and resume skills.
    input_schema: JobSearchInputModel      # or inline dict { resume: ResumeModel, ... }
    output_schema: "array[JobListingModel]"
    rate_limit: { calls: 30, per_seconds: 60 }
    auth: { type: api_key, env: JOB_API_KEY }
    retry: { max_attempts: 2 }
```

| Field | Description |
| :--- | :--- |
| `impl` | Dotted path (`module.func`) or a function name resolved from `modules.tools` (defaults to the tool key). |
| `description` | Description exposed to the LLM. |
| `input_schema` | Name of a Pydantic `BaseModel` in `modules.data_models` (bound as `args_schema`), or an inline field dict. |
| `rate_limit` | `{ calls, per_seconds }`, enforced with `TokenBucketRateLimiter`. |
| `auth` | `{ type, env }` reads `os.environ[env]` and injects it as `api_key`, `token`, `credential`, `auth`, or into `**kwargs` when present on the function signature. |
| `retry` | Per-tool retry policy passed to `retry_call`. |

### `evaluators`

Evaluation functions that run after agent nodes.

```yaml
evaluators:
  schema_validator:   { impl: evaluators.schema_validator,  type: automatic }
  grounding_checker:  { impl: evaluators.grounding_checker, type: automatic }
  user_feedback_loop: { impl: evaluators.user_feedback,     type: human_feedback, source: state.suggestions }
```

Built-in evaluators return metrics such as `json_parse_errors`, `field_hallucination_rate`, and `suggestion_acceptance_rate`.

### `agents`

```yaml
agents:
  suggestions_agent:
    description: Proposes ATS-focused improvements to the parsed resume.
    type: chat                  # chat (default) | react
    tools: [document_editor]
    input:  { resume: state.parsed_resume }
    output: state.suggestions
    harness:
      system_instruction: "Provide high-impact structural adjustments."
      llm_config: { model: gemini-3.5-flash, temperature: 0.7 }
      fallback_llm_config: { model: gemini-3.5-flash, temperature: 0.0 }
      context_policy:
        pre_filters:
          - target: input.resume
            action: strip_keys
            args: [contact_info, references]
        few_shot:
          enabled: true
          vector_store: memory.long_term.successful_parses
          max_examples: 2
          selection_strategy: cosine_similarity
        window_management:
          max_tokens: 4096
          overflow_strategy: summarize_oldest
          max_messages: 10              # used by sliding_window
          preserve_recent_count: 1      # used by summarize_oldest
      input_template: |
        Parsed Resume Data:
        {{ input.resume }}
    retry: { max_attempts: 1 }
    optimization:
      evaluator: user_feedback_loop
      max_mutation_depth: 3
      fallback_harness: default
      metrics:
        - { key: suggestion_acceptance_rate, target: ">= 0.75" }
      on_underperformance:
        action: mutate_context      # mutate_harness | mutate_context
        strategies: [few_shot_refresh, dynamic_prompt_template_evolution]
        few_shot_source: memory.long_term.accepted_suggestions
```

**Inputs and outputs.** `input` maps local template keys to `state.<field>` references; `output` names the target `state.<field>`.

**Prompt rendering.**
- `harness.input_template` is a Jinja2 string rendered with `state` and `input` (after `pre_filters`).
- `harness.input_transformer` is either `{ type: "template", value: "..." }` or `{ type: "function", module: "...", function: "...", args: {...} }`.

**Context policy.**
- `pre_filters` run before prompt rendering. `strip_keys` removes the listed keys from `input.<key>` without mutating state.
- `few_shot`, when enabled with a store, queries `(vector_store.split(".")[-1], "global")` for up to `max_examples` and appends them to the system prompt.
- `window_management` supports `summarize_oldest`, `drop_oldest`, `truncate`, and `sliding_window`.

**Optimization.**
- `metrics[].target` is a comparison (`==`, `!=`, `>=`, `<=`, `>`, `<`) followed by a number, or a bare numeric literal.
- Built-in `strategies`: `cot_prompt_injection`, `negative_prompting_injection`, `temperature_cooling`, `few_shot_refresh`, `dynamic_prompt_template_evolution`.
- `mutate_context` is restricted to `input_template*` and `context_policy*`; `mutate_harness` may change any field in `mutable_fields`.
- Harness changes are versioned in a `HarnessRegistry`, gated by `min_samples_before_mutation`, optional offline evaluation and human approval, and rolled back automatically on regression.

### `nodes`

#### Standard agent and periodic nodes

```yaml
- id: node_job_search
  agent: job_search_agent
  type: periodic              # omit for a standard agent node
  interval_s: 3600
  gate_conditions:
    - "state.preferences != null"
    - "state.parsed_resume != null"
  trigger_when: "state.resume_file != null"
  routing:
    - { when: "len(state.job_listings) > 0", to: node_job_analyzer }
```

If `trigger_when` or any `gate_conditions` entry is false, the node skips execution and returns `{}`.

#### Swarm nodes

A swarm compiles into two graph nodes: `<id>` (fans out `Send` objects) and `<id>_worker` (processes one item and writes to `collect.to`, where the state reducer aggregates results). Downstream `routing` / `to` edges attach to `<id>_worker`.

```yaml
- id: node_job_analyzer
  agent: job_analyzer_agent
  type: swarm
  count: 10
  strategy: map_split
  distribution_method: round_robin
  iterator:
    loop_over: state.job_listings
    filter: "item.id not in state.job_analysis.job_ids"
    yield_to: state.single_job_listing
  collect:
    from: state.single_job_analysis
    to: state.job_analysis
    reducer: append
  routing:
    - { when: "state.job_analysis != null", to: node_application_review }
```

#### Consumer nodes

Iterate sequentially over `iterator.loop_over`, bind each item to `iterator.yield_to`, and collect outputs into the agent's output key.

```yaml
- id: node_application
  agent: application_agent
  type: consumer
  iterator:
    loop_over: state.approved_job_analysis
    yield_to: state.approved_job_analysis
  routing:
    - { when: "true", to: end }
```

#### Human-in-the-loop nodes

| Variant | Trigger | Behavior |
| :--- | :--- | :--- |
| **Input** | no `agent`, no `review` | Interrupts with `input_type` (`file`, `text`, or `form`) and writes the response (or file contents) to `input_state`. |
| **Review** | `review` block present | Filters `review.source` with `review.pre_filter`, auto-approves items matching `auto_approve_if`, interrupts only for the rest, and writes approved items to `review.select_to`. |
| **Agent-assisted** | `agent` set | Interrupts for a decision (`dict`, `bool`, or natural-language feedback delegated to the agent). |

#### Routing and error handling

- **`routing`**: ordered list of `{ when: <expr_or_dict>, to: <node_id | [node_ids] | end> }`. `to` may be a list for parallel fan-out, or `end` / `END` / `null` for LangGraph `END`. An optional `default` sets the fallback (defaults to `END`).
- **`on_error`**: `{ action: route_to_healer, then: <node_id> }`. After retries are exhausted, execution moves via `Command(goto=<then>, update={"_error": ...})`. With self-healing enabled, `SelfHealer` also versions any proposed graph patch and adds `"_heal": {"version": ..., "patch": ...}` to the update.

### Condition expression language

Strings in `when`, `trigger_when`, `gate_conditions`, `iterator.filter`, `review.pre_filter`, and `auto_approve_if` are evaluated safely by an AST visitor (`utils.StateConditionEvaluator`), without `eval()`.

- **Identifiers and attributes:** `state.field.nested`, `item.score`, `len(state.job_listings)`
- **Literals:** `true` / `false` / `null` (and Python `True` / `False` / `None`), numbers, strings
- **Operators:** `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not in`, `and`, `or`, `not`, unary `-`

---

## Modules and Project Layout

### Public modules

| Module | Exports |
| :--- | :--- |
| `builder` | `generate_agent_yaml(spec, ...)`, `generate_companion_modules(yaml_text, spec)`, `validate_agent_yaml(yaml_text)`, `ensure_agent_yaml_path(path_str)`, `main()` |
| `resilience` | `retry_call`, `TokenBucketRateLimiter`, `GraphRegistry`, `GraphVersion`, `apply_graph_patch`, `SelfHealer` |
| `optimization` | `HarnessRegistry`, `Window`, `optimize`, `apply_optimization`, `validate_optimization`, `target_met`, `STRATEGIES` |
| `memory` | `create_memory(memory_config)`, `apply_context_policy(messages, context_policy, llm)` |
| `evaluators` | `schema_validator`, `grounding_checker`, `user_feedback` |
| `utils` | `resolve_module`, `deep_merge`, `apply_pre_filters`, `extract_input_context`, `render_input_prompt`, `ConfigResolver`, `evaluate_condition` |

### Repository layout

```text
meta-agent/
├── src/meta_agent_madcoderme/
│   ├── __init__.py      # Public exports and __version__
│   ├── compiler.py      # CLI compiler and end-to-end .agent.yaml runner
│   ├── builder.py       # LLM-powered generator for .agent.yaml and companion modules
│   ├── graph.py         # StateGraph builder, node executors, swarm fan-out, routing, error/healer/eval wrappers
│   ├── state.py         # TypedDict state builder, built-in reducers, scope metadata
│   ├── nodes.py         # NetworkAgent, harness merging, LLM/ReAct initialization, human input nodes
│   ├── tools.py         # Dynamic tool loader, Pydantic args_schema binding, rate limiting, auth injection
│   ├── memory.py        # Checkpointer/store factory (in_memory, postgres) and context window management
│   ├── resilience.py    # Retries, rate limiter, trace-reading SelfHealer, versioned GraphRegistry
│   ├── evaluators.py    # Built-in automatic and human-feedback evaluators
│   ├── optimization.py  # Target parsing, mutation strategies, HarnessRegistry, guarded optimize()
│   └── utils.py         # Module resolver, deep_merge, pre-filters, Jinja2 rendering, AST condition evaluator
├── examples/
│   └── basic_graph.py   # Minimal LangGraph sanity example
├── tests/
│   ├── jobsearch.agent.yaml
│   ├── data_models.py
│   ├── tool_defs.py
│   └── test_*.py
├── pyproject.toml
└── .env.example
```

---

## Safety and Security Notes

- **Irreversible actions need a human gate.** Tools with external side effects (submitting applications, sending messages, writing to third-party systems) should sit behind a `human_in_the_loop` review node, and tools should not be retried blindly (`retry: { max_attempts: 1 }`).
- **Treat generated code as untrusted until reviewed.** Output from `meta-agent-build` and any self-healing graph patch should be reviewed before production use.
- **Protect sensitive data.** Use `context_policy.pre_filters` to strip fields such as contact information from prompts, and keep API keys in the environment, never in YAML.
- **Self-improvement is guarded, not unattended.** Keep `human_approval_for: [system_instruction]`, set `min_samples_before_mutation`, and consider `require_offline_eval: true` for production.
- **Routing conditions are sandboxed.** They are parsed by an AST visitor and cannot execute arbitrary code.

---

## Development and Testing

The test suite uses `unittest` and mocks LLM and store calls, so no API keys are required.

```bash
python -m unittest discover -s tests -q
```

---

## Troubleshooting

| Symptom | Likely cause and fix |
| :--- | :--- |
| `UserWarning: ... uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored` | The configured model ignores `temperature`. Remove it from the harness, and note that `temperature_cooling` has no effect on such models. |
| Agent output in state is a raw list of content blocks instead of a model instance | Enable schema enforcement (`output_validation.enforce_schema`) and bind the agent's output to a Pydantic model in `data_models`. |
| `Module not found` for `data_models` / `tool_defs` | Check `modules:` names. Modules are resolved from standard imports, `tests.<name>`, and `<name>.py` in the working directory or `tests/`. |
| Node skipped without running | A `trigger_when` or `gate_conditions` expression evaluated false. Check the referenced state fields. |
| Condition raises an error | Only the documented constructs are supported (see [Condition expression language](#condition-expression-language)). |

---

## Contributing

Issues and pull requests are welcome. Please run the test suite before submitting changes.

## License

Add your license here (for example, MIT) and include a `LICENSE` file in the repository.