# Meta Agent

Meta Agent enables people and AI agents to build production-grade agent harnesses and multi-agent networks simply by defining a high-level design in YAML (`*.agent.yaml`). At its core is a **compiler** (`compiler.py` / `graph.py`) that turns a declarative `schema_version: "1.0"` specification directly into an executable [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`—wiring typed state and reducers, parallel swarm fan-out (`Send`), sequential consumers, human-in-the-loop interrupts, AST-evaluated routing, context policies, resilience, trace-driven self-healing with graph versioning, evaluators, and guarded harness optimization without hand-coding boilerplate orchestration.

For rapid scaffolding from natural language, Meta Agent also provides a CLI builder (`builder.py`) that generates a valid `.agent.yaml` file from a high-level description using an LLM, ready to be compiled and executed immediately.

---

## Requirements

- **Python**: `>= 3.11`
- **Dependencies** (declared in `pyproject.toml`):
  - `langgraph>=1.0`
  - `langchain>=1.0`
  - `langchain-core>=1.0`
  - `langchain-google-genai>=4.0`
  - `pydantic>=2.0`
  - `pyyaml>=6.0`
  - `jinja2>=3.1`
- **API Keys**: A `GEMINI_API_KEY` (or credentials for your configured LLM provider) when executing live model calls.

---

## Installation & Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` (and optional tool credentials such as `JOB_API_KEY` or `APPLICATION_API_KEY` if your tools use `auth.env`).

---

## Usage: High-Level Design → Compile → Run

### 1. Generate a High-Level `.agent.yaml` Design & Companion Modules (`builder.py`)

If you want to scaffold a complete, execution-ready network from high-level requirements instead of authoring YAML and boilerplate Python by hand, use `builder.py` (or the installed `meta-agent-build` console script). It calls a configured LLM to generate and validate the `.agent.yaml` file, then generates and syntax-checks the companion Python modules (`data_models.py`, `tool_defs.py`, and custom `hooks.py` / `evaluators.py` if referenced) in the same directory:

```bash
# Single-command generation of .agent.yaml + companion Python modules
meta-agent-build "Build a customer support triage network with intake, specialist swarm analysis, human approval, and reply delivery" \
  --name "Support Triage" \
  --agents "intake_agent,specialist_agent,reply_agent" \
  --features "swarm,human_in_the_loop,self_healing,optimization" \
  -o support/support_triage.agent.yaml

# Or interactive wizard mode (prompts for name, goal, agents, tools, features, output file)
python -m meta_agent_madcoderme.builder -i
```

Pass `--yaml-only` if you only want the `.agent.yaml` file without generating companion Python modules. You can also choose the builder LLM with `--provider`, `--model`, and `--temperature` (defaults: `google-genai`, `gemini-3.6-flash`, `0.2`), or call `meta_agent_madcoderme.generate_agent_yaml(spec)` and `meta_agent_madcoderme.generate_companion_modules(yaml_text, spec)` from Python.

### 2. Compile & Run a Network (`compiler.py`)

The compiler (`meta_agent_madcoderme.compiler` / `meta-agent`) is the core entrypoint. Pass any `.agent.yaml` file as a positional argument:

```bash
meta-agent tests/jobsearch.agent.yaml
# or directly:
python -m meta_agent_madcoderme.compiler tests/jobsearch.agent.yaml
```

The compiler:
- Validates that the file path ends with `.agent.yaml` and exists on disk.
- Compiles typed state, tools, agents (with inherited `harness_defaults`), memory stores, resilience/self-healing wrappers, evaluators, and the LangGraph `StateGraph`.
- Prints the compiled ASCII graph (`print_ascii()`).
- Executes the workflow and prompts interactively whenever a human-in-the-loop `__interrupt__` is reached, then prints the final execution trace.

### 3. Using the Compiler Programmatically

You can embed the compiler directly in any Python application to load high-level `.agent.yaml` designs and run them:

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

# 3. Invoke with initial state and thread config
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

### 4. Standalone Subsystem Modules

- **`builder.py`**: `generate_agent_yaml(spec, ...)`, `validate_agent_yaml(yaml_text)`, `ensure_agent_yaml_path(path_str)`, and CLI `main()`.
- **`resilience.py`**: `retry_call(operation, config)`, `TokenBucketRateLimiter(calls, per_seconds)`, `GraphRegistry`, `GraphVersion`, `apply_graph_patch`, and `SelfHealer`.
- **`optimization.py`**: `HarnessRegistry`, `Window`, `optimize(...)`, `apply_optimization(...)`, `validate_optimization(spec)`, `target_met`, and `STRATEGIES`.
- **`memory.py`**: `create_memory(memory_config)` and `apply_context_policy(messages, context_policy, llm)`.
- **`evaluators.py`**: Built-in `schema_validator`, `grounding_checker`, and `user_feedback`.
- **`utils.py`**: `resolve_module`, `deep_merge`, `apply_pre_filters`, `extract_input_context`, `render_input_prompt`, `ConfigResolver`, and `evaluate_condition`.

---

## Project Layout

```text
meta-agent/
├── src/meta_agent_madcoderme/
│   ├── __init__.py      # Public package exports & __version__
│   ├── compiler.py      # Core CLI compiler & end-to-end .agent.yaml runner
│   ├── builder.py       # LLM-powered CLI generator for .agent.yaml + companion Python modules
│   ├── graph.py         # StateGraph builder, node executors, swarm Send fan-out, routing, error/healer/eval wrappers
│   ├── state.py         # TypedDict state builder, built-in reducers (append, dedupe_by_id), scope metadata
│   ├── nodes.py         # NetworkAgent class, harness merging, LLM/React agent initialization, human input nodes
│   ├── tools.py         # Dynamic tool loader, Pydantic args_schema binding, rate limiting, auth env injection
│   ├── memory.py        # Checkpointer/Store factory (in_memory, postgres) & context window management
│   ├── resilience.py    # Retries, TokenBucketRateLimiter, trace-reading SelfHealer & versioned GraphRegistry
│   ├── evaluators.py    # Built-in automatic & human-feedback evaluators
│   ├── optimization.py  # Metric target parsing, mutation strategies, HarnessRegistry, guarded optimize()
│   └── utils.py         # Module resolver, deep_merge, pre-filters, Jinja2 prompt rendering, AST condition evaluator
├── examples/
│   └── basic_graph.py   # Minimal LangGraph sanity example
├── tests/
│   ├── jobsearch.agent.yaml
│   ├── data_models.py
│   ├── tool_defs.py
│   └── test_*.py        # Unit test suite
├── pyproject.toml
└── .env.example
```

---

## YAML Agent Definition Schema (`schema_version: "1.0"`)

A complete reference example is available in [tests/jobsearch.agent.yaml](tests/jobsearch.agent.yaml). Below is the full supported schema specification.

### Top-Level Keys

| Key | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `schema_version` | `string` | Yes | Schema version identifier (e.g. `"1.0"`). |
| `name` | `string` | No | Human-readable network name. |
| `description` | `string` | No | Description of the agent network. |
| `entry_node` | `string` | Yes* | ID of the starting node in `nodes`. (*Or set `is_starting_node: true` on a node.) |
| `modules` | `object` | No | Mapping of logical module roles to Python module names. |
| `runtime` | `object` | No | Global memory, resilience, and self-improvement policies. |
| `harness_defaults` | `object` | No | Default agent harness configuration inherited by all agents via deep merge. |
| `state` | `object` | Yes | Shared and worker-scoped state field definitions. |
| `tools` | `object` | No | Tool definitions bound to agents by name. |
| `evaluators` | `object` | No | Evaluator definitions referenced by agent `optimization` blocks. |
| `agents` | `object` | Yes | Agent definitions (models, harnesses, I/O mappings, retries, optimization). |
| `nodes` | `list` | Yes | Graph nodes, execution types, iterators, and routing rules. |

---

### `modules`

Maps logical roles to Python modules. `utils.resolve_module()` searches standard imports, `tests.<name>`, and `<name>.py` in the working directory or `tests/`.

```yaml
modules:
  data_models: data_models   # Pydantic models referenced by state & tool input_schema
  tools: tool_defs           # Callable tool implementations
  evaluators: evaluators     # Evaluator functions
  hooks: hooks               # Optional custom state reducer functions
```

---

### `runtime`

Global runtime policies for memory, resilience, and guarded self-improvement.

```yaml
runtime:
  memory:
    stores:
      short_term:
        type: in_memory        # in_memory | postgres
        connection_string: ""  # Required when type: postgres
        scope: run
      long_term:
        type: in_memory        # in_memory | postgres
        connection_string: ""  # Required when type: postgres
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

Additional schema blocks such as `concurrency` and `monitoring` may also be declared under `runtime` for configuration completeness. When `runtime.self_healing.enabled` is `true`, `create_graph()` attaches a `GraphRegistry` (`workflow.__graph_registry__`) and `SelfHealer` (`workflow.__self_healer__`) configured with `healer_model`. On unhandled node errors, `SelfHealer` reads execution traces (`_trace` and `messages`), proposes a JSON graph patch (`node_updates`, `add_nodes`, `remove_nodes`, `reroute`, `goto`), records a new `GraphVersion` (`0 -> 1 -> ...`, preserving prior versions as `superseded` and supporting `rollback()`), and routes via LangGraph `Command(goto=..., update={"_error": ..., "_heal": ...})`.

---

### `harness_defaults`

Provides default configuration inherited by every agent. Agent-level `harness` blocks are combined using `utils.deep_merge(harness_defaults, agent_harness)`.

```yaml
harness_defaults:
  id: default
  llm_config:
    provider: google-genai     # Also accepts model_provider
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

---

### `state`

Defines fields on the compiled `NetworkState` `TypedDict`. A `messages` field annotated with LangGraph's `add_messages` reducer is automatically included.

```yaml
state:
  resume_file:            { type: file }
  parsed_resume:          { type: ResumeModel }
  job_listings:           { type: "array[JobListingModel]", reducer: dedupe_by_id }
  job_analysis:           { type: "array[JobAnalysisModel]", reducer: append }
  single_job_listing:     { type: JobListingModel, scope: worker }
```

| Field Property | Values / Behavior |
| :--- | :--- |
| `type` | Name of a class in `modules.data_models` (e.g. `ResumeModel`), `array[<Type>]` (mapped to `List[<Type>]`), or any unmapped primitive name (defaults to `Any`). |
| `reducer` | `append` (appends items/lists), `dedupe_by_id` (deduplicates by `id` key or `.id` attribute), or the name of a custom reducer function in `modules.hooks`. |
| `scope` | `global` (default) or `worker`. Recorded on `NetworkState.__state_scopes__`. |

---

### `tools`

Defines callable tools that agents can reference by name in their `tools` list.

```yaml
tools:
  job_listings_fetcher:
    type: api_caller
    impl: tools.job_listings_fetcher
    description: Fetches job listings matching preferences and resume skills.
    input_schema: JobSearchInputModel   # Or inline dict { resume: ResumeModel, ... }
    output_schema: "array[JobListingModel]"
    rate_limit: { calls: 30, per_seconds: 60 }
    auth: { type: api_key, env: JOB_API_KEY }
    retry: { max_attempts: 2 }
```

| Field | Description |
| :--- | :--- |
| `impl` | Dotted path (`module.func`) or function name resolved from `modules.tools` (defaults to the tool key name). |
| `description` | Tool description exposed to the LLM. |
| `input_schema` | String name of a Pydantic `BaseModel` in `modules.data_models` (bound as `args_schema`), or an inline field dict. |
| `rate_limit` | `{ calls, per_seconds }` wrapped with `TokenBucketRateLimiter`. |
| `auth` | `{ type, env }` reads `os.environ[env]` and injects it into `api_key`, `token`, `credential`, `auth`, or `**kwargs` if present on the target function signature. |
| `retry` | Per-tool retry policy passed to `retry_call`. |

---

### `evaluators`

Defines evaluation functions executed after agent nodes run.

```yaml
evaluators:
  schema_validator:   { impl: evaluators.schema_validator,  type: automatic }
  grounding_checker:  { impl: evaluators.grounding_checker, type: automatic }
  user_feedback_loop: { impl: evaluators.user_feedback,     type: human_feedback, source: state.suggestions }
```

Built-in evaluators in `evaluators.py` return metric dictionaries such as `json_parse_errors`, `field_hallucination_rate`, and `suggestion_acceptance_rate`.

---

### `agents`

Defines LLM-backed or ReAct agents.

```yaml
agents:
  suggestions_agent:
    description: Proposes ATS-focused improvements to the parsed resume.
    type: chat                  # chat (default) | react
    tools: [document_editor]    # List of tool keys from `tools:`
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
          overflow_strategy: summarize_oldest   # summarize_oldest | drop_oldest | truncate | sliding_window
          max_messages: 10                      # Used by sliding_window
          preserve_recent_count: 1              # Used by summarize_oldest
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
        action: mutate_context                  # mutate_harness | mutate_context
        strategies: [few_shot_refresh, dynamic_prompt_template_evolution]
        few_shot_source: memory.long_term.accepted_suggestions
```

#### Agent Details

- **`input` & `output`**: `input` maps local template keys to `state.<field>` references; `output` specifies the target `state.<field>`.
- **`harness.input_template` / `harness.input_transformer`**:
  - `input_template`: Jinja2 string rendered with `state` and `input` (after `pre_filters` are applied).
  - `input_transformer`: Dict `{ type: "template", value: "..." }` or `{ type: "function", module: "...", function: "...", args: {...} }`.
- **`harness.context_policy`**:
  - `pre_filters`: Applied before prompt rendering. `action: strip_keys` removes listed keys from `input.<key>` without mutating state.
  - `few_shot`: When `enabled: true` and a `store` is configured, queries `(vector_store.split(".")[-1], "global")` up to `max_examples` and appends them to the system prompt.
  - `window_management`: Supports `summarize_oldest`, `drop_oldest`, `truncate`, and `sliding_window`.
- **`optimization`**:
  - `metrics[].target`: Comparison expression (`==`, `!=`, `>=`, `<=`, `>`, `<`) followed by a number, or a bare numeric literal.
  - `strategies`: Built-in mutators in `optimization.STRATEGIES` are `cot_prompt_injection`, `negative_prompting_injection`, `temperature_cooling`, `few_shot_refresh`, and `dynamic_prompt_template_evolution`.
  - `mutate_context` restricts mutations to `input_template*` and `context_policy*`; `mutate_harness` allows all fields in `mutable_fields`.

---

### `nodes`

Defines graph topology, execution patterns, interrupts, error handling, and conditional routing.

#### Node Types

1. **Standard Agent Node (`type: agent`, default)** and **Periodic Node (`type: periodic`)**:
   ```yaml
   - id: node_job_search
     agent: job_search_agent
     type: periodic
     interval_s: 3600
     gate_conditions:
       - "state.preferences != null"
       - "state.parsed_resume != null"
     trigger_when: "state.resume_file != null"
     routing:
       - { when: "len(state.job_listings) > 0", to: node_job_analyzer }
   ```
   If `trigger_when` or any entry in `gate_conditions` evaluates to `False`, the node skips execution and returns `{}`.

2. **Swarm Node (`type: swarm`)**:
   Compiles into two graph nodes: `<id>` (a fan-out node returning LangGraph `Send` objects to `<id>_worker`) and `<id>_worker` (executes one item and emits its result under `collect.to` for state reducer aggregation). Downstream `routing` / `to` edges are attached to `<id>_worker`.
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

3. **Consumer Node (`type: consumer`)**:
   Sequentially iterates over `iterator.loop_over`, binding each item to `iterator.yield_to`, and collects outputs into the agent's output state key.
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

4. **Human-in-the-Loop Nodes (`type: human_in_the_loop`)**:
   Supports three variants:
   - **Input node** (no `agent` and no `review`): Interrupts with `input_type` (`file` | `text` | `form`) and writes the response (or file contents for `file`) to `input_state`.
   - **Review node** (`review` block present): Filters `review.source` using `review.pre_filter`, auto-approves items matching `auto_approve_if`, and calls `interrupt()` only for remaining items. Writes approved items to `review.select_to`.
   - **Agent-assisted decision node** (`agent` set): Interrupts for human decision (`dict`, `bool`, or natural language feedback delegated to `agent`).

#### Routing & Error Handling

- **`routing`**: List of `{ when: <expr_or_dict>, to: <node_id | [node_ids] | end> }`, evaluated in order. `to` may be a list for parallel fan-out, or `end` / `END` / `null` for LangGraph `END`. Optional `default` sets the fallback target (defaults to `END`).
- **`on_error`**: `{ action: route_to_healer, then: <node_id> }`. If a node raises an unhandled exception after retries are exhausted, execution routes via `Command(goto=<then>, update={"_error": ...})`. When `runtime.self_healing.enabled` is `true`, `SelfHealer` inspects the trace, versions any proposed graph patch in `GraphRegistry`, and includes `"_heal": {"version": ..., "patch": ...}` in the state update.

---

### Condition Expression Language

String conditions in `when`, `trigger_when`, `gate_conditions`, `iterator.filter`, `review.pre_filter`, and `auto_approve_if` are safely evaluated using an AST visitor (`utils.StateConditionEvaluator`) without `eval()`.

Supported constructs:
- **Identifiers & attributes**: `state.field.nested`, `item.score`, `len(state.job_listings)`
- **Literals**: YAML `true` / `false` / `null` (and Python `True` / `False` / `None`), numbers, strings
- **Operators**: `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not in`, `and`, `or`, `not`, unary `-`

---

## Testing

The test suite uses Python's `unittest` framework and mocks LLM/store calls so no API keys are required:

```bash
python -m unittest discover -s tests -q
```
