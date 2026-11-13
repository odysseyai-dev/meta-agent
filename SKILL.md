---
name: meta-agent
description: Build, compile, run, and debug multi-agent networks defined in YAML (`*.agent.yaml`) with the py-meta-agent package, which compiles a declarative spec into an executable LangGraph StateGraph. Use when the user wants to design an agent network, write or fix an .agent.yaml file, create companion modules (data_models.py, tool_defs.py, hooks.py, evaluators.py), run the `meta-agent` or `meta-agent-build` CLI, or configure swarms, human-in-the-loop nodes, self-healing, or harness optimization.
---

# Meta Agent

Meta Agent (`pip install py-meta-agent`, import name `meta_agent_madcoderme`) turns a `schema_version: "1.0"` YAML file into a LangGraph `StateGraph`. The user describes state, tools, agents, and nodes; the compiler handles orchestration. Requires Python >= 3.11 and an LLM credential (e.g. `GEMINI_API_KEY`) for live runs.

## Workflow

1. **Clarify the network** (only what's missing): goal, agents, tools with side effects, where a human must approve, whether work should run in parallel.
2. **Design state first.** List the fields that flow between agents, with types and reducers.
3. **Write the `.agent.yaml`.** The filename must end in `.agent.yaml`.
4. **Write companion modules** the YAML references (see below).
5. **Run it:** `meta-agent path/to/network.agent.yaml`. This compiles, prints the ASCII graph, and prompts at each human-in-the-loop interrupt.
6. **Inspect the trace and the final state**, then iterate.

To scaffold from a description instead of hand-writing:

```bash
meta-agent-build "<goal>" --name "<Name>" --agents "a_agent,b_agent" \
  --features "swarm,human_in_the_loop,self_healing,optimization" -o dir/name.agent.yaml
```

Use `--yaml-only` to skip companion modules. Always review generated YAML and tool code before running anything with external side effects.

## Minimal skeleton

```yaml
schema_version: "1.0"
name: Example
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
      input_template: |
        Question: {{ input.question }}

nodes:
  - id: ask
    type: human_in_the_loop
    input_type: text            # file | text | form
    input_state: question
    routing:
      - { when: "state.question != null", to: respond }
  - id: respond
    agent: answerer
    routing:
      - { when: "true", to: end }
```

## Schema essentials

Top-level keys: `schema_version`, `name`, `description`, `entry_node`, `modules`, `runtime`, `harness_defaults`, `state`, `tools`, `evaluators`, `agents`, `nodes`. Required: `schema_version`, `state`, `agents`, `nodes`, and an entry node (`entry_node`, or `is_starting_node: true` on a node).

- **state**: `type` is a Pydantic class from `modules.data_models`, `array[<Type>]`, or a primitive. `reducer`: `append`, `dedupe_by_id`, or a custom function in `modules.hooks`. `scope: worker` for per-worker fields in swarms.
- **tools**: `impl` (dotted path or name from `modules.tools`), `description`, `input_schema` (a BaseModel name or inline dict), optional `rate_limit`, `auth: { type, env }`, `retry`.
- **agents**: `type: chat` (default) or `react`; `tools`; `input` (local key -> `state.<field>`); `output` (`state.<field>`); `harness`; `retry`; `optimization`.
- **harness_defaults**: deep-merged under every agent's `harness`, so agents override only what differs (e.g. just `llm_config: { temperature: 0.0 }`).
- **harness keys**: `system_instruction`, `llm_config`, `fallback_llm_config`, `input_template` (Jinja2 with `state` and `input`), `input_transformer`, `output_validation`, `context_policy` (`pre_filters`, `few_shot`, `window_management`).

### Node types

| Type | Use for | Key fields |
| :--- | :--- | :--- |
| agent (default) | Run one agent | `agent`, `routing`, `trigger_when`, `gate_conditions` |
| `periodic` | Re-run on an interval | `interval_s`, `gate_conditions` |
| `swarm` | Parallel map over a list | `iterator.loop_over/filter/yield_to`, `collect.from/to/reducer`, `count` |
| `consumer` | Sequential loop over a list | `iterator.loop_over/yield_to` |
| `human_in_the_loop` | Input, review, or decision | input: `input_type`, `input_state`; review: `review.source/pre_filter/select_to`, `auto_approve_if`; decision: `agent` |

Routing: ordered `routing: [{ when: <expr>, to: <node | [nodes] | end> }]`, optional `default`. A list for `to` fans out in parallel. `on_error: { action: route_to_healer, then: <node> }` handles failures.

A swarm compiles into `<id>` (fan-out via `Send`) and `<id>_worker`; downstream edges attach to the worker.

### Condition language

Used in `when`, `trigger_when`, `gate_conditions`, `iterator.filter`, `review.pre_filter`, `auto_approve_if`. Evaluated by an AST visitor, not `eval()`. Supported: `state.a.b`, `item.score`, `len(...)`, literals (`true`/`false`/`null`, numbers, strings), `== != > < >= <= in "not in" and or not`, unary `-`. Nothing else.

## Companion modules

Resolved from `modules:` (standard imports, `tests.<name>`, or `<name>.py` in the working directory or `tests/`).

- `data_models.py`: Pydantic models for state and tool schemas.
- `tool_defs.py`: tool functions; the name must match `impl` (or the tool key).
- `hooks.py`: custom reducers referenced by `reducer:`.
- `evaluators.py`: evaluators referenced under `evaluators:`. Built-ins: `schema_validator`, `grounding_checker`, `user_feedback`.

Auth: a tool's `auth.env` is read from the environment and injected as `api_key`, `token`, `credential`, `auth`, or `**kwargs` if the function signature has one of them.

## Design rules (apply when writing or reviewing YAML)

1. **One writer per state key where possible.** If two agents write the same key (e.g. an updater writing `state.suggestions`), they overwrite each other. Give each output its own field.
2. **Type every state field and bind agent outputs to a model.** Without schema enforcement, outputs end up as raw LLM content blocks in state, which breaks pre-filters and conditions like `item.score >= 80`.
3. **Swarm workers must use `scope: worker` fields** for the per-item input and output, and use `collect` to reduce results into a shared field. Never have workers write one shared field directly.
4. **Gate irreversible actions** (submitting, sending, paying) behind a `human_in_the_loop` review node; set `retry: { max_attempts: 1 }` on those agents; give the tool a `dry_run` option.
5. **Every state field a node reads must be populated by an earlier node.** Check that something (often a human input node) sets each field used in `gate_conditions`.
6. **Give every branch an exit.** Route rejections and timeouts to `end` or a fallback; use `default:`.
7. **Strip sensitive fields** with `context_policy.pre_filters` (`strip_keys`) before prompts.
8. **Give each agent only the context it needs.** For example, an analyzer needs both the item and the resume to score fit.
9. **Don't repeat config.** Put shared model and policy settings in `harness_defaults`; override per agent only where they differ.
10. **Don't set `temperature` for models that ignore it.** The runtime warns when a model uses fixed sampling defaults, and `temperature_cooling` is then a no-op.

## Runtime features (when asked)

- **resilience**: `default_retry` (attempts, backoff, jitter, `retry_on`), `default_timeout_s`, `on_unhandled_error: route_to_healer`.
- **self_healing**: a healer model reads the execution trace and proposes a JSON graph patch (`node_updates`, `add_nodes`, `remove_nodes`, `reroute`, `goto`). Patches are versioned in a `GraphRegistry` with rollback. Strategies: `retry`, `fallback_harness`, `fallback_model`, `modify_graph`, `escalate_to_human`.
- **self_improving**: guardrails include `require_versioning`, `auto_rollback_on_regression`, `regression_tolerance`, `max_mutation_depth`, `human_approval_for`, `min_samples_before_mutation`, `mutable_fields`, `require_offline_eval`. Per-agent `optimization` has `evaluator`, `metrics` (`target` like `"== 0"`, `"< 0.05"`, `">= 0.75"`), and `on_underperformance` with `action` (`mutate_harness` | `mutate_context`) and `strategies` (`cot_prompt_injection`, `negative_prompting_injection`, `temperature_cooling`, `few_shot_refresh`, `dynamic_prompt_template_evolution`). `mutate_context` only touches `input_template*` and `context_policy*`.
- **memory**: `runtime.memory.stores` with `short_term` and `long_term`; type `in_memory` or `postgres` (needs `connection_string`). Agents reference namespaces as `memory.long_term.<namespace>`.

## Running programmatically

```python
import yaml
from langgraph.types import Command
from meta_agent_madcoderme import create_states, create_tools, create_agents, create_memory, create_graph

cfg = yaml.safe_load(open("network.agent.yaml", encoding="utf-8"))
modules, runtime = cfg.get("modules", {}), cfg.get("runtime", {})
state = create_states(cfg.get("state", {}), modules_config=modules)
tools = create_tools(cfg.get("tools", {}), modules_config=modules)
agents = create_agents(cfg.get("agents", {}), tools, harness_defaults=cfg.get("harness_defaults", {}))
checkpointer, store = create_memory(runtime.get("memory"))
wf = create_graph(network_state=state, agent_list=agents, network_nodes=cfg.get("nodes", []),
                  memory_config=runtime.get("memory"), tools=tools, entry_node=cfg.get("entry_node"),
                  runtime_config=runtime, evaluator_config=cfg.get("evaluators", {}), modules_config=modules)
app = wf.compile(checkpointer=checkpointer, store=store)

cfg_run = {"configurable": {"thread_id": "session-1"}}
init = {k: [] if k == "messages" else None for k in state.__annotations__}
result = app.invoke(init, cfg_run)
while "__interrupt__" in result:                       # human-in-the-loop
    info = result["__interrupt__"][0].value
    result = app.invoke(Command(resume=input(f"{info.get('prompt')}: ")), config=cfg_run)
```

## Debugging checklist

| Symptom | Check |
| :--- | :--- |
| Node never runs | `trigger_when` / `gate_conditions` false; the referenced state field is never set. |
| Run ends early with nothing applied | A routing `when` (or review `pre_filter`) evaluated false or errored on unparsed state; confirm fields are typed models, not raw text. |
| State key holds the wrong content | Two agents write the same key; check each agent's `output`. |
| Agent got huge or odd prompts | Raw content blocks (with metadata) are being stringified into `input_template`; enforce structured output. |
| `Module not found` | `modules:` names and file location (working directory or `tests/`). |
| Tool auth fails | `auth.env` variable not set, or the function signature lacks a parameter to receive it. |
| Condition errors | Uses a construct outside the supported list. |
| Swarm results missing | Missing `collect`, or workers wrote to a non-worker field. |

## Tests

```bash
python -m unittest discover -s tests -q
```

Tests mock LLM and store calls; no API keys needed. A full reference network is `tests/jobsearch.agent.yaml`.