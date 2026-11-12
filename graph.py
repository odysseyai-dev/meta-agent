# graph.py

from typing import List, Dict, Any, Optional, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Send, Command

from memory import apply_context_policy
from nodes import NetworkAgent, create_human_node
from utils import ConfigResolver, render_input_prompt, evaluate_condition, apply_pre_filters, extract_input_context, deep_merge, resolve_module
from resilience import retry_call
from optimization import apply_optimization

def make_node_executor(agent_obj: NetworkAgent, node_info: Optional[Dict[str, Any]] = None):
    """Executes a single standard or periodic agent node."""
    node_info = node_info or {}

    def node_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        # 1. Trigger conditions
        trigger = node_info.get('trigger_when') or node_info.get('trigger_cond') or getattr(agent_obj, 'trigger', None)
        if trigger and not evaluate_condition(trigger, state):
            print(f"[SKIP] Node {node_info.get('id', agent_obj.id)} trigger condition '{trigger}' not met.")
            return {}

        # 2. Gate conditions (for periodic / gated nodes)
        gate_conditions = node_info.get('gate_conditions', [])
        for gate_cond in gate_conditions:
            if not evaluate_condition(gate_cond, state):
                print(f"[SKIP] Node {node_info.get('id', agent_obj.id)} gate condition '{gate_cond}' not met.")
                return {}

        llm = agent_obj.get_agent()

        # 3. Memory context — long-term global memories
        long_term_memories = []
        if store:
            user_namespace = ("memories", "global")
            mem_items = store.search(user_namespace)
            long_term_memories = [item.value for item in mem_items]

        # 3b. Few-shot retrieval from long-term store
        few_shot_memories = []
        context_policy = agent_obj.harness.get("context_policy", {}) or {}
        few_shot_config = context_policy.get("few_shot")
        if isinstance(few_shot_config, dict) and few_shot_config.get("enabled", False) and store:
            raw_ns = few_shot_config.get("vector_store", "memory.long_term.few_shot")
            # Map "memory.long_term.<namespace>" → (<namespace>, "global")
            ns_key = raw_ns.split(".")[-1]
            few_shot_namespace = (ns_key, "global")
            max_examples = few_shot_config.get("max_examples", 3)
            try:
                few_shot_items = store.search(few_shot_namespace, limit=max_examples)
                few_shot_memories = [item.value for item in few_shot_items]
            except Exception:
                pass  # store may not support similarity search; degrade gracefully

        # 4. System prompt
        raw_sys_spec = agent_obj.harness.get("system_instruction") or agent_obj.harness.get("system_prompt", "")
        resolved_sys_prompt = ConfigResolver.resolve_value(raw_sys_spec, state=state, store=store)
        if long_term_memories:
            resolved_sys_prompt += f"\n\nLong-term Context/Memories: {long_term_memories}"
        if few_shot_memories:
            few_shot_lines = "\n".join(f"Example {i+1}: {ex}" for i, ex in enumerate(few_shot_memories))
            resolved_sys_prompt += f"\n\nFew-shot Examples:\n{few_shot_lines}"

        messages = [SystemMessage(content=resolved_sys_prompt)]
        existing_messages = state.get("messages") or []
        if existing_messages:
            messages.extend(existing_messages)

        # 5. Apply context pre-filters then build user prompt
        pre_filters = context_policy.get("pre_filters", []) or []
        input_context = extract_input_context(agent_obj, state)
        if pre_filters:
            filtered_context = apply_pre_filters({"input": input_context}, pre_filters)
            # Rebuild a filtered state-like scope so render_input_prompt uses cleaned data
            filtered_state = dict(state)
            # Merge filtered input keys back so templates referencing state.* still work
            for k, v in filtered_context.get("input", {}).items():
                # Map input key back to its original state key via agent input mapping
                raw_input = getattr(agent_obj, "input", {}) or {}
                state_ref = raw_input.get(k, "")
                if isinstance(state_ref, str) and state_ref.startswith("state."):
                    state_key = state_ref[len("state."):]
                    filtered_state[state_key] = v
        else:
            filtered_state = state

        transformer_spec = agent_obj.harness.get("input_transformer")
        if transformer_spec:
            user_prompt_text = ConfigResolver.resolve_value(transformer_spec, state=filtered_state, store=store)
        else:
            user_prompt_text = render_input_prompt(agent_obj, filtered_state)

        current_user_msg = HumanMessage(content=user_prompt_text)
        messages.append(current_user_msg)
        messages = apply_context_policy(messages, agent_obj.harness.get("context_policy", {}), llm)

        # 6. Tool binding
        bound_llm = llm.bind_tools(agent_obj.tools) if (llm and agent_obj.tools) else llm
        node_new_messages = [current_user_msg]

        # 7. LLM invocation with fallback support
        try:
            response = bound_llm.invoke(messages) if bound_llm else HumanMessage(content="No LLM bound.")
        except Exception as primary_err:
            fallback = agent_obj.get_fallback_agent()
            if fallback and fallback != bound_llm:
                print(f"[FALLBACK] Node {node_info.get('id', agent_obj.id)} invoking fallback model due to: {primary_err}")
                bound_fallback = fallback.bind_tools(agent_obj.tools) if agent_obj.tools else fallback
                response = bound_fallback.invoke(messages)
            else:
                raise primary_err

        messages.append(response)
        node_new_messages.append(response)

        # 8. Tool loop
        while getattr(response, "tool_calls", None):
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                t_id = tool_call["id"]

                matching_tool = next((t for t in agent_obj.tools if t.name == t_name), None)
                if matching_tool:
                    try:
                        tool_result = matching_tool.invoke(t_args)
                    except Exception as e:
                        tool_result = f"Error executing tool {t_name}: {str(e)}"
                else:
                    tool_result = f"Error: Tool {t_name} is not registered on this agent."

                tool_msg = ToolMessage(content=str(tool_result), tool_call_id=t_id)
                messages.append(tool_msg)
                node_new_messages.append(tool_msg)

            response = bound_llm.invoke(messages)
            messages.append(response)
            node_new_messages.append(response)

        out_key = agent_obj.get_output_key() if hasattr(agent_obj, "get_output_key") else "output"
        out_content = response.content if hasattr(response, "content") else str(response)

        return {
            out_key: out_content,
            "messages": node_new_messages
        }

    return node_fn

def make_swarm_fanout(agent_obj: NetworkAgent, node_info: Dict[str, Any]):
    """Returns a fan-out function that emits Send() for each item — enables LangGraph parallel execution."""
    iterator_cfg = node_info.get("iterator", {})
    collect_cfg = node_info.get("collect", {})

    node_id = node_info["id"]
    worker_node_id = f"{node_id}_worker"

    loop_target_expr = iterator_cfg.get("loop_over", "").replace("state.", "")
    yield_to_key = iterator_cfg.get("yield_to", "item").replace("state.", "")
    filter_expr = iterator_cfg.get("filter")

    collect_from_key = collect_cfg.get("from", agent_obj.get_output_key()).replace("state.", "")
    collect_to_key = collect_cfg.get("to", agent_obj.get_output_key()).replace("state.", "")

    def fanout_fn(state: Dict[str, Any]) -> List[Send]:
        raw_items = state.get(loop_target_expr, [])
        if not raw_items:
            return []

        items = raw_items if isinstance(raw_items, list) else [raw_items]
        sends = []
        for item in items:
            if filter_expr:
                env = {"item": item, "state": state}
                if not evaluate_condition(filter_expr, env):
                    continue
            worker_state = dict(state)
            worker_state[yield_to_key] = item
            # Pass bookkeeping keys so worker knows where to write its result
            worker_state["__swarm_collect_from__"] = collect_from_key
            worker_state["__swarm_collect_to__"] = collect_to_key
            sends.append(Send(worker_node_id, worker_state))
        return sends

    return fanout_fn


def make_swarm_worker(agent_obj: NetworkAgent, node_info: Dict[str, Any]):
    """Executes one swarm work unit and writes result to the collect-to key (reduced by append)."""
    single_executor = make_node_executor(agent_obj, node_info)
    collect_cfg = node_info.get("collect", {})
    default_collect_to = collect_cfg.get("to", agent_obj.get_output_key()).replace("state.", "")

    def worker_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        collect_from_key = state.get("__swarm_collect_from__") or agent_obj.get_output_key()
        collect_to_key = state.get("__swarm_collect_to__") or default_collect_to

        result = single_executor(state, store=store)

        # Pull the worker's output and emit it under the collect-to key so the
        # state reducer (append / dedupe) merges it into the shared list.
        worker_output = result.get(collect_from_key) or result.get(agent_obj.get_output_key())
        out: Dict[str, Any] = {}
        if worker_output is not None:
            out[collect_to_key] = worker_output
        if "messages" in result:
            out["messages"] = result["messages"]
        return out

    return worker_fn


# Keep make_swarm_executor as a sequential fallback used directly in unit tests
def make_swarm_executor(agent_obj: NetworkAgent, node_info: Dict[str, Any]):
    """Sequential fallback executor — used in unit tests and when Send-based routing is not wired."""
    single_executor = make_node_executor(agent_obj, node_info)
    iterator_cfg = node_info.get("iterator", {})
    collect_cfg = node_info.get("collect", {})

    loop_target_expr = iterator_cfg.get("loop_over", "").replace("state.", "")
    yield_to_key = iterator_cfg.get("yield_to", "item").replace("state.", "")
    filter_expr = iterator_cfg.get("filter")

    collect_from_key = collect_cfg.get("from", agent_obj.get_output_key()).replace("state.", "")
    collect_to_key = collect_cfg.get("to", agent_obj.get_output_key()).replace("state.", "")

    def swarm_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        raw_items = state.get(loop_target_expr, [])
        if not raw_items:
            return {}

        items = raw_items if isinstance(raw_items, list) else [raw_items]
        collected = []
        all_messages = []

        for item in items:
            if filter_expr:
                env = {"item": item, "state": state}
                if not evaluate_condition(filter_expr, env):
                    continue

            worker_state = dict(state)
            worker_state[yield_to_key] = item

            worker_res = single_executor(worker_state, store=store)
            if collect_from_key in worker_res:
                collected.append(worker_res[collect_from_key])
            elif agent_obj.get_output_key() in worker_res:
                collected.append(worker_res[agent_obj.get_output_key()])

            if "messages" in worker_res:
                all_messages.extend(worker_res["messages"])

        return {
            collect_to_key: collected,
            "messages": all_messages
        }

    return swarm_fn

def make_consumer_executor(agent_obj: NetworkAgent, node_info: Dict[str, Any]):
    """Executes a consumer node over approved/ready items sequentially."""
    single_executor = make_node_executor(agent_obj, node_info)
    iterator_cfg = node_info.get("iterator", {})
    loop_target_expr = iterator_cfg.get("loop_over", "").replace("state.", "")
    yield_to_key = iterator_cfg.get("yield_to", "item").replace("state.", "")
    out_key = agent_obj.get_output_key()

    def consumer_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        raw_items = state.get(loop_target_expr, [])
        if not raw_items:
            return {}

        items = raw_items if isinstance(raw_items, list) else [raw_items]
        results = []
        all_messages = []

        for item in items:
            worker_state = dict(state)
            worker_state[yield_to_key] = item
            worker_res = single_executor(worker_state, store=store)
            if out_key in worker_res:
                results.append(worker_res[out_key])
            if "messages" in worker_res:
                all_messages.extend(worker_res["messages"])

        return {
            out_key: results,
            "messages": all_messages
        }

    return consumer_fn

def make_review_node(node_info: Dict[str, Any]):
    """Executes human review node with pre-filtering, auto-approval checks, and approval interrupts."""
    review_cfg = node_info.get("review", {})
    source_expr = review_cfg.get("source", "").replace("state.", "")
    select_to_key = review_cfg.get("select_to", "approved").replace("state.", "")
    pre_filter = review_cfg.get("pre_filter")
    auto_approve_if = node_info.get("auto_approve_if")

    def review_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        raw_source = state.get(source_expr, [])
        items = raw_source if isinstance(raw_source, list) else ([raw_source] if raw_source is not None else [])

        # Apply pre-filter
        filtered_items = []
        for item in items:
            if pre_filter:
                env = {"item": item, "state": state}
                if evaluate_condition(pre_filter, env):
                    filtered_items.append(item)
            else:
                filtered_items.append(item)

        if not filtered_items:
            return {select_to_key: []}

        # Check auto-approval
        auto_approved = []
        needs_review = []
        for item in filtered_items:
            if auto_approve_if:
                env = {"item": item, "state": state}
                if evaluate_condition(auto_approve_if, env):
                    auto_approved.append(item)
                    continue
            needs_review.append(item)

        # Trigger interrupt if human review is needed
        if needs_review:
            user_response = interrupt({
                "node_id": node_info.get("id"),
                "input_type": "review",
                "items_for_review": needs_review,
                "prompt": node_info.get("prompt", f"Please review items for {select_to_key}")
            })
            if isinstance(user_response, list):
                approved_total = auto_approved + user_response
            elif user_response is True:
                approved_total = auto_approved + needs_review
            elif isinstance(user_response, dict) and "approved" in user_response:
                approved_total = auto_approved + user_response["approved"]
            else:
                approved_total = auto_approved
        else:
            approved_total = auto_approved

        return {select_to_key: approved_total}

    return review_fn

def make_human_agent_node(agent_obj: NetworkAgent, node_info: Dict[str, Any]):
    """Executes human-in-the-loop node assisted by an agent (e.g. suggestions editor)."""
    agent_executor = make_node_executor(agent_obj, node_info)
    out_key = agent_obj.get_output_key()

    def human_agent_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        target_data = state.get(out_key) or state.get("suggestions")
        user_response = interrupt({
            "node_id": node_info.get("id"),
            "input_type": "decision",
            "data": target_data,
            "prompt": node_info.get("prompt", f"Please review and decide on {node_info.get('id')}")
        })

        if isinstance(user_response, dict):
            current = dict(target_data) if isinstance(target_data, dict) else {}
            current.update(user_response)
            return {out_key: current}
        elif isinstance(user_response, bool):
            current = dict(target_data) if isinstance(target_data, dict) else {}
            current["is_approved"] = user_response
            return {out_key: current}

        worker_state = dict(state)
        worker_state["human_response"] = user_response
        return agent_executor(worker_state, store=store)

    return human_agent_fn

def create_graph(
    network_state: Any,
    agent_list: List[NetworkAgent], 
    network_nodes: List[Dict[str, Any]], 
    memory_config: Optional[Dict[str, Any]] = None,
    tools: Optional[Dict[str, BaseTool]] = None,
    entry_node: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    modules_config: Optional[Dict[str, str]] = None):
    
    workflow = StateGraph(state_schema=network_state)
    agent_map = {a.id: a for a in agent_list}
    runtime_config = runtime_config or {}
    resilience_config = runtime_config.get("resilience", {}) or {}
    default_retry = resilience_config.get("default_retry", {}) or {}
    global_guardrails = (runtime_config.get("self_improving", {}) or {}).get("guardrails", {}) or {}
    evaluator_config = evaluator_config or {}
    modules_config = modules_config or {}

    evaluator_module = resolve_module(modules_config.get("evaluators", "evaluators"))

    def run_evaluator(agent_obj, output, state):
        spec = agent_obj.get_optimization() if hasattr(agent_obj, "get_optimization") else {}
        evaluator_name = spec.get("evaluator")
        if not evaluator_name:
            return {}
        evaluator_spec = evaluator_config.get(evaluator_name, {})
        impl = evaluator_spec.get("impl", evaluator_name) if isinstance(evaluator_spec, dict) else evaluator_name
        if isinstance(impl, str) and "." in impl:
            module_name, function_name = impl.rsplit(".", 1)
            module = resolve_module(module_name) or evaluator_module
        else:
            module, function_name = evaluator_module, impl
        evaluator = getattr(module, function_name, None) if module else None
        if not evaluator:
            return {}
        try:
            result = evaluator(output=output, state=state)
        except TypeError:
            result = evaluator(output, state)
        return result if isinstance(result, dict) else {"score": result}

    def wrap_node(node_fn, node_info, agent_obj):
        retry_policy = dict(default_retry)
        if agent_obj:
            retry_policy = deep_merge(retry_policy, agent_obj.get_retry_config() or {})
        error_spec = node_info.get("on_error", {}) or {}
        error_target = error_spec.get("then")

        def guarded_node(state: Dict[str, Any], store: BaseStore = None):
            def execute():
                result = node_fn(state, store=store)
                if agent_obj and isinstance(result, dict):
                    metrics = run_evaluator(agent_obj, result.get(agent_obj.get_output_key()), state)
                    if metrics:
                        apply_optimization(
                            agent_obj,
                            metrics,
                            sample_count=state.get("__sample_count__", 0),
                            global_guardrails=global_guardrails,
                        )
                return result

            try:
                return retry_call(execute, retry_policy)
            except Exception as error:
                if error_target:
                    update = {"__error__": {"node": node_info.get("id"), "type": type(error).__name__, "message": str(error)}}
                    return Command(goto=error_target, update=update)
                raise

        return guarded_node

    if entry_node:
        workflow.set_entry_point(entry_node)

    # First pass: collect all swarm node IDs so downstream routing can resolve
    # "{node_id}" → "{node_id}_worker" (the actual graph node that finishes last).
    swarm_node_ids = {
        n["id"] for n in network_nodes
        if n.get("type") == "swarm" and agent_map.get(n.get("agent"))
    }

    def resolve_target(t):
        if t is None or t in ("end", "END", END):
            return END
        return t

    for n in network_nodes:
        node_id = n["id"]
        node_type = n.get("type", "agent")
        agent_id = n.get("agent")
        agent_obj = agent_map.get(agent_id) if agent_id else None

        # --- Register node(s) ---
        if node_type == "swarm" and agent_obj:
            worker_node_id = f"{node_id}_worker"
            print(f"Adding swarm fan-out node {node_id} and worker {worker_node_id} with agent {agent_id}")
            # Fan-out: returns List[Send] — LangGraph dispatches workers in parallel
            workflow.add_conditional_edges(node_id, make_swarm_fanout(agent_obj, n))
            # Dummy identity node so the fan-out source exists in the graph
            workflow.add_node(node_id, lambda state: {})
            # Worker node: processes one item
            workflow.add_node(worker_node_id, wrap_node(make_swarm_worker(agent_obj, n), n, agent_obj))
        elif node_type == "consumer" and agent_obj:
            print(f"Adding consumer node {node_id} with agent {agent_id}")
            workflow.add_node(node_id, wrap_node(make_consumer_executor(agent_obj, n), n, agent_obj))
        elif node_type == "human_in_the_loop":
            if "review" in n:
                print(f"Adding review human-in-the-loop node {node_id}")
                workflow.add_node(node_id, wrap_node(make_review_node(n), n, None))
            elif agent_obj:
                print(f"Adding agent-assisted human-in-the-loop node {node_id}")
                workflow.add_node(node_id, wrap_node(make_human_agent_node(agent_obj, n), n, agent_obj))
            else:
                print(f"Adding input human-in-the-loop node {node_id}")
                workflow.add_node(node_id, wrap_node(create_human_node(n), n, None))
        else:
            if agent_obj:
                print(f"Adding agent node {node_id} with agent {agent_id}")
                workflow.add_node(node_id, wrap_node(make_node_executor(agent_obj, n), n, agent_obj))

        if n.get('is_starting_node', False):
            workflow.set_entry_point(node_id)

        # --- Static edges ---
        if n.get("to"):
            for target_node in n['to']:
                dest = END if target_node in ("end", "END", None) else target_node
                # For swarm: wire from worker node, not the fan-out shell
                edge_source = f"{node_id}_worker" if node_type == "swarm" and agent_obj else node_id
                print(f"Adding edge from {edge_source} to {dest}")
                workflow.add_edge(edge_source, dest)

        # --- Conditional routing ---
        if n.get('routing'):
            def create_router(rules, default_target):
                def router(state: Dict[str, Any], store: BaseStore = None):
                    for rule in rules:
                        cond = rule.get("when") if "when" in rule else rule.get("condition")
                        if isinstance(cond, dict):
                            res = ConfigResolver.resolve_value(cond, state=state, store=store)
                            if bool(res):
                                target = rule.get('to')
                                if isinstance(target, list):
                                    return [resolve_target(t) for t in target]
                                return resolve_target(target)
                        elif isinstance(cond, str):
                            if evaluate_condition(cond, state):
                                target = rule.get('to')
                                if isinstance(target, list):
                                    return [resolve_target(t) for t in target]
                                return resolve_target(target)
                    return resolve_target(default_target)
                return router

            default_dest = n.get('default', END)
            path_map = {}
            for rule in n['routing']:
                target = rule.get('to')
                if isinstance(target, list):
                    for t in target:
                        dest = resolve_target(t)
                        path_map[dest] = dest
                else:
                    dest = resolve_target(target)
                    path_map[dest] = dest

            def_dest = resolve_target(default_dest)
            path_map[def_dest] = def_dest
            path_map[END] = END

            # For swarm: routing comes FROM the worker node (the last node to finish),
            # not the fan-out shell node.
            routing_source = f"{node_id}_worker" if node_type == "swarm" and agent_obj else node_id
            print(f"Adding conditional routing from {routing_source} → {list(path_map.keys())}")
            workflow.add_conditional_edges(
                routing_source,
                create_router(n['routing'], default_dest),
                path_map
            )

    return workflow
