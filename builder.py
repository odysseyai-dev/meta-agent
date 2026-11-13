"""CLI network builder that turns high-level requirements into a .agent.yaml design using an LLM."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
import yaml


SCHEMA_SYSTEM_PROMPT = """You are an expert agent-network architect for the Meta-Agent compiler (schema_version: "1.0").
Given a user's high-level description, generate a complete, valid `.agent.yaml` configuration file.

Output ONLY valid YAML (no prose outside YAML). Follow the schema_version "1.0" specification:
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
"""


BUILTIN_STATE_TYPES = {
    "str", "string", "int", "integer", "float", "number", "bool", "boolean",
    "dict", "object", "list", "any", "Any", "file",
}
BUILTIN_REDUCERS = {"append", "dedupe_by_id"}
BUILTIN_EVALUATORS = {"schema_validator", "grounding_checker", "user_feedback"}

MODULE_SYSTEM_PROMPT = """You are an expert Python engineer building companion modules for a Meta-Agent `.agent.yaml` network.
Given the `.agent.yaml` configuration and required symbol list, output ONLY a valid JSON object mapping Python filenames to complete, runnable Python source code.

Rules:
- Keys must be filenames matching `modules` in the YAML (e.g. `"data_models.py"`, `"tool_defs.py"`, and if custom evaluators/hooks are used, `"evaluators.py"` / `"hooks.py"`).
- In `data_models.py`, define every referenced Pydantic `BaseModel` class using `pydantic.BaseModel` and `Field` with sensible typed fields and defaults.
- In `tool_defs.py`, implement every referenced tool function with clear type annotations, docstrings, and working logic (accepting `api_key: str = ""` or `**kwargs` if the tool declares `auth`).
- In `hooks.py` (if custom reducers are used), implement `(existing, new) -> list` reducer functions.
- In `evaluators.py` (if custom evaluators are used), implement `(output=..., state=...) -> dict` functions.
- Output ONLY valid JSON (no markdown prose outside the JSON).
"""


def _unwrap_array_type(type_expr: Any) -> str:
    text = str(type_expr or "Any").strip()
    while text.startswith("array[") and text.endswith("]"):
        text = text[6:-1].strip()
    return text


def extract_required_symbols(parsed_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect parsed `.agent.yaml` to identify companion Python modules and symbols needed."""
    modules_cfg = parsed_cfg.get("modules") or {}
    dm_module = str(modules_cfg.get("data_models", "data_models")).split(".")[-1]
    tools_module = str(modules_cfg.get("tools", "tool_defs")).split(".")[-1]
    eval_module = str(modules_cfg.get("evaluators", "evaluators")).split(".")[-1]
    hooks_module = str(modules_cfg.get("hooks", "hooks")).split(".")[-1]

    models: List[str] = []
    seen_models = set()

    def add_model(name: str) -> None:
        clean = _unwrap_array_type(name)
        if clean and clean not in BUILTIN_STATE_TYPES and clean not in seen_models:
            seen_models.add(clean)
            models.append(clean)

    for field_cfg in (parsed_cfg.get("state") or {}).values():
        if isinstance(field_cfg, dict):
            add_model(field_cfg.get("type", "Any"))
        elif isinstance(field_cfg, str):
            add_model(field_cfg)

    custom_reducers: List[str] = []
    for field_cfg in (parsed_cfg.get("state") or {}).values():
        if isinstance(field_cfg, dict) and field_cfg.get("reducer"):
            r_name = str(field_cfg["reducer"])
            if r_name not in BUILTIN_REDUCERS and r_name not in custom_reducers:
                custom_reducers.append(r_name)

    tool_specs: List[Dict[str, Any]] = []
    for tool_name, tool_info in (parsed_cfg.get("tools") or {}).items():
        if not isinstance(tool_info, dict):
            continue
        impl = str(tool_info.get("impl", "") or "")
        fn_name = impl.rsplit(".", 1)[-1] if "." in impl else (impl or tool_name)
        raw_in = tool_info.get("input_schema") or tool_info.get("input")
        if isinstance(raw_in, str):
            add_model(raw_in)
        elif isinstance(raw_in, dict):
            for v in raw_in.values():
                add_model(v)
        if isinstance(tool_info.get("output_schema"), str):
            add_model(tool_info["output_schema"])
        tool_specs.append({
            "name": tool_name,
            "func_name": fn_name,
            "description": tool_info.get("description", f"Implementation for {tool_name}."),
            "input_schema": raw_in,
            "has_auth": bool(tool_info.get("auth")),
        })

    custom_evaluators: List[str] = []
    for ev_name, ev_info in (parsed_cfg.get("evaluators") or {}).items():
        impl = ev_info.get("impl", ev_name) if isinstance(ev_info, dict) else ev_name
        fn_name = str(impl).rsplit(".", 1)[-1]
        if fn_name not in BUILTIN_EVALUATORS and fn_name not in custom_evaluators:
            custom_evaluators.append(fn_name)

    return {
        "module_files": {
            "data_models": f"{dm_module}.py",
            "tools": f"{tools_module}.py",
            "evaluators": f"{eval_module}.py",
            "hooks": f"{hooks_module}.py",
        },
        "models": models,
        "tools": tool_specs,
        "custom_reducers": custom_reducers,
        "custom_evaluators": custom_evaluators,
    }


def scaffold_companion_modules(parsed_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Deterministically generate valid companion Python modules from parsed `.agent.yaml`."""
    symbols = extract_required_symbols(parsed_cfg)
    files: Dict[str, str] = {}
    mod_files = symbols["module_files"]

    # 1. data_models.py
    dm_lines = [
        '"""Pydantic data models for the generated agent network."""',
        "",
        "from typing import Any, Dict, List, Optional",
        "from pydantic import BaseModel, Field",
        "",
    ]
    if symbols["models"]:
        for cls_name in symbols["models"]:
            dm_lines.extend([
                f"class {cls_name}(BaseModel):",
                f'    """Auto-generated model for {cls_name}."""',
                "    id: Optional[str] = None",
                "    data: Dict[str, Any] = Field(default_factory=dict)",
                "",
            ])
    else:
        dm_lines.extend([
            "class NetworkPayload(BaseModel):",
            "    data: Dict[str, Any] = Field(default_factory=dict)",
            "",
        ])
    files[mod_files["data_models"]] = "\n".join(dm_lines)

    # 2. tool_defs.py
    if symbols["tools"]:
        tool_lines = [
            '"""Tool implementations for the generated agent network."""',
            "",
            "from typing import Any, Dict",
            "",
        ]
        for t in symbols["tools"]:
            params: List[str] = []
            raw_in = t.get("input_schema")
            if isinstance(raw_in, dict) and raw_in:
                for p_name in raw_in.keys():
                    params.append(f"{p_name}: Any = None")
            else:
                params.append("payload: Any = None")
            if t.get("has_auth"):
                params.append('api_key: str = ""')
            params.append("**kwargs: Any")
            sig = ", ".join(params)
            desc = str(t.get("description", "")).replace('"""', '\\"\\"\\"')
            tool_lines.extend([
                f"def {t['func_name']}({sig}) -> Dict[str, Any]:",
                f'    """{desc}"""',
                f'    return {{"tool": "{t["name"]}", "status": "ok", "kwargs": kwargs}}',
                "",
            ])
        files[mod_files["tools"]] = "\n".join(tool_lines)

    # 3. hooks.py (only if custom reducers are declared)
    if symbols["custom_reducers"]:
        hook_lines = [
            '"""Custom state reducers for the generated agent network."""',
            "",
            "from typing import Any, List, Optional",
            "",
        ]
        for r_name in symbols["custom_reducers"]:
            hook_lines.extend([
                f"def {r_name}(existing: Optional[List[Any]], new: Any) -> List[Any]:",
                "    base = list(existing or [])",
                "    if new is None:",
                "        return base",
                "    return base + (new if isinstance(new, list) else [new])",
                "",
            ])
        files[mod_files["hooks"]] = "\n".join(hook_lines)

    # 4. evaluators.py (only if custom evaluators outside built-ins are declared)
    if symbols["custom_evaluators"]:
        ev_lines = [
            '"""Custom evaluators for the generated agent network."""',
            "",
            "from typing import Any, Dict, Optional",
            "",
        ]
        for ev_name in symbols["custom_evaluators"]:
            ev_lines.extend([
                f"def {ev_name}(output: Any = None, state: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:",
                "    valid = output is not None",
                '    return {"score": 1.0 if valid else 0.0, "valid": valid}',
                "",
            ])
        files[mod_files["evaluators"]] = "\n".join(ev_lines)

    return files


def _extract_json_object(raw_text: Any) -> Optional[Dict[str, str]]:
    text = str(raw_text or "").strip()
    fenced = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(k): str(v) for k, v in parsed.items() if str(k).endswith(".py") and isinstance(v, str)}


def generate_companion_modules(
    yaml_text: str,
    spec: Dict[str, Any],
    *,
    llm: Any = None,
    provider: str = "google-genai",
    model: str = "gemini-3.6-flash",
    temperature: float = 0.2,
) -> Dict[str, str]:
    """Generate syntax-validated companion Python modules for a `.agent.yaml` definition."""
    parsed_cfg = validate_agent_yaml(yaml_text)
    fallback_files = scaffold_companion_modules(parsed_cfg)
    symbols = extract_required_symbols(parsed_cfg)

    generated: Optional[Dict[str, str]] = None
    try:
        client = llm if llm is not None else _build_default_llm(provider, model, temperature)
        prompt = (
            f"Network description: {spec.get('description', '')}\n\n"
            f"Required symbols:\n{json.dumps(symbols, indent=2)}\n\n"
            f"Agent YAML:\n```yaml\n{yaml_text}\n```"
        )
        response = client.invoke([
            SystemMessage(content=MODULE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw_content = getattr(response, "content", response)
        if isinstance(raw_content, list):
            raw_content = "\n".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in raw_content
            )
        generated = _extract_json_object(raw_content)
    except Exception:
        generated = None

    final_files: Dict[str, str] = {}
    for fname, fallback_code in fallback_files.items():
        candidate = (generated or {}).get(fname)
        if candidate:
            try:
                ast.parse(candidate)
                final_files[fname] = candidate if candidate.endswith("\n") else candidate + "\n"
                continue
            except SyntaxError:
                pass
        ast.parse(fallback_code)
        final_files[fname] = fallback_code

    # Include any extra valid .py modules the LLM produced
    for fname, code in (generated or {}).items():
        if fname not in final_files:
            try:
                ast.parse(code)
                final_files[fname] = code if code.endswith("\n") else code + "\n"
            except SyntaxError:
                pass

    return final_files


def write_network_bundle(
    output_path: Path,
    yaml_text: str,
    modules: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Write the `.agent.yaml` file and any companion Python modules to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_text, encoding="utf-8")
    written = [output_path]

    for filename, code in (modules or {}).items():
        mod_path = output_path.parent / Path(filename).name
        mod_path.write_text(code, encoding="utf-8")
        written.append(mod_path)

    return written


def build_generation_prompt(spec: Dict[str, Any]) -> str:
    """Format high-level user requirements into an LLM generation prompt."""
    lines = ["Generate a complete schema_version: '1.0' `.agent.yaml` file for the following design:"]
    if spec.get("name"):
        lines.append(f"- Network Name: {spec['name']}")
    if spec.get("description"):
        lines.append(f"- Goal & Workflow Description: {spec['description']}")
    if spec.get("agents"):
        lines.append(f"- Desired Agents / Roles: {spec['agents']}")
    if spec.get("tools"):
        lines.append(f"- Tools / Integrations: {spec['tools']}")
    if spec.get("features"):
        lines.append(f"- Requested Capabilities / Node Patterns: {spec['features']}")
    lines.append("\nReturn valid YAML only.")
    return "\n".join(lines)


def extract_yaml_content(raw_text: Any) -> str:
    """Extract raw YAML text from an LLM response, stripping optional markdown code fences."""
    text = str(raw_text or "").strip()
    fenced = re.search(r"```(?:ya?ml)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text + ("\n" if text and not text.endswith("\n") else "")


def validate_agent_yaml(yaml_text: str) -> Dict[str, Any]:
    """Parse and validate that generated YAML contains the required compiler sections."""
    parsed = yaml.safe_load(yaml_text)
    if not isinstance(parsed, dict):
        raise ValueError("Generated YAML must parse to a mapping at the top level.")

    required = ["schema_version", "state", "agents", "nodes"]
    missing = [k for k in required if k not in parsed or parsed[k] is None]
    if missing:
        raise ValueError(f"Generated YAML is missing required section(s): {', '.join(missing)}")

    nodes = parsed.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Generated YAML `nodes` must be a non-empty list.")

    has_entry = bool(parsed.get("entry_node")) or any(
        isinstance(n, dict) and n.get("is_starting_node") for n in nodes
    )
    if not has_entry:
        raise ValueError("Generated YAML must specify `entry_node` or a node with `is_starting_node: true`.")

    return parsed


def ensure_agent_yaml_path(path_str: str) -> Path:
    """Normalize an output path so it always uses the `.agent.yaml` suffix required by compiler.py."""
    cleaned = path_str.strip()
    if not cleaned:
        cleaned = "network.agent.yaml"
    if cleaned.endswith(".agent.yaml"):
        return Path(cleaned)
    for suffix in (".yaml", ".yml"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return Path(f"{cleaned}.agent.yaml")


def _build_default_llm(provider: str, model: str, temperature: float) -> Any:
    from langchain.chat_models import init_chat_model

    return init_chat_model(model=model, model_provider=provider, temperature=temperature)


def generate_agent_yaml(
    spec: Dict[str, Any],
    *,
    llm: Any = None,
    provider: str = "google-genai",
    model: str = "gemini-3.6-flash",
    temperature: float = 0.2,
) -> str:
    """Generate and validate `.agent.yaml` text from a high-level network specification."""
    client = llm if llm is not None else _build_default_llm(provider, model, temperature)
    response = client.invoke([
        SystemMessage(content=SCHEMA_SYSTEM_PROMPT),
        HumanMessage(content=build_generation_prompt(spec)),
    ])
    raw_content = getattr(response, "content", response)
    if isinstance(raw_content, list):
        raw_content = "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    yaml_text = extract_yaml_content(raw_content)
    validate_agent_yaml(yaml_text)
    return yaml_text


def prompt_interactive_spec(
    existing: Optional[Dict[str, Any]] = None,
    input_fn: Callable[[str], str] = input,
) -> Dict[str, Any]:
    """Interactively prompt the user for high-level agent network details."""
    spec = dict(existing or {})
    if not spec.get("name"):
        spec["name"] = input_fn("Network name (e.g. Support Triage): ").strip()
    if not spec.get("description"):
        spec["description"] = input_fn("Describe the workflow and goal: ").strip()
    if not spec.get("agents"):
        spec["agents"] = input_fn("Agents/roles (optional, comma-separated): ").strip()
    if not spec.get("tools"):
        spec["tools"] = input_fn("Tools/integrations (optional, comma-separated): ").strip()
    if not spec.get("features"):
        spec["features"] = input_fn("Features (optional: swarm, human_in_the_loop, self_healing, optimization): ").strip()
    if not spec.get("output"):
        default_slug = re.sub(r"[^a-z0-9]+", "_", (spec.get("name") or "network").lower()).strip("_") or "network"
        chosen = input_fn(f"Output file [{default_slug}.agent.yaml]: ").strip()
        spec["output"] = chosen or f"{default_slug}.agent.yaml"
    return spec


def main(argv: Optional[List[str]] = None, *, llm: Any = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Meta-Agent Network Builder: create a .agent.yaml design from high-level requirements using an LLM."
    )
    parser.add_argument("description", nargs="?", default=None, help="High-level description of the network to build")
    parser.add_argument("-n", "--name", default=None, help="Name of the agent network")
    parser.add_argument("-o", "--output", default=None, help="Output .agent.yaml file path")
    parser.add_argument("--agents", default=None, help="Comma-separated agent roles or descriptions")
    parser.add_argument("--tools", default=None, help="Comma-separated tool names or descriptions")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated capabilities (e.g. swarm,human_in_the_loop,self_healing,optimization)",
    )
    parser.add_argument("--provider", default="google-genai", help="LLM provider (default: google-genai)")
    parser.add_argument("--model", default="gemini-3.6-flash", help="LLM model (default: gemini-3.6-flash)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument(
        "--yaml-only",
        action="store_true",
        help="Only generate the .agent.yaml file without companion Python modules",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Prompt interactively for design details")
    args = parser.parse_args(argv)

    spec: Dict[str, Any] = {
        "name": args.name,
        "description": args.description,
        "agents": args.agents,
        "tools": args.tools,
        "features": args.features,
        "output": args.output,
    }

    if args.interactive or not spec.get("description"):
        spec = prompt_interactive_spec(spec)

    if not spec.get("description"):
        raise ValueError("A high-level network description is required to generate a .agent.yaml file.")

    output_name = spec.get("output") or (
        re.sub(r"[^a-z0-9]+", "_", (spec.get("name") or "network").lower()).strip("_") or "network"
    )
    output_path = ensure_agent_yaml_path(output_name)

    yaml_text = generate_agent_yaml(
        spec,
        llm=llm,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    modules: Dict[str, str] = {}
    if not args.yaml_only:
        modules = generate_companion_modules(
            yaml_text,
            spec,
            llm=llm,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
        )

    written_paths = write_network_bundle(output_path, yaml_text, modules)

    print(f"Generated agent network definition: {output_path}")
    for mod_path in written_paths[1:]:
        print(f"Generated companion module: {mod_path}")
    print(f"Compile & run with: meta-agent {output_path}  (or: python compiler.py {output_path})")
    return output_path


if __name__ == "__main__":
    main()
