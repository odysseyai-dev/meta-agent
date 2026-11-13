"""Small, dependency-free resilience helpers used by graph and tool execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import random
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional


def classify_error(error: BaseException) -> set[str]:
    text = str(error).lower()
    kinds = {"transient_error"}
    if isinstance(error, (TimeoutError, TimeoutError)) or "timeout" in text:
        kinds.add("timeout")
    if "rate" in text and "limit" in text:
        kinds.add("rate_limit")
    if "schema" in text or "validation" in text or "json" in text:
        kinds.add("schema_violation")
    return kinds


def retry_call(
    operation: Callable[[], Any],
    config: Optional[Dict[str, Any]] = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Any:
    """Run an operation according to a YAML retry policy.

    ``max_attempts`` includes the initial call. A policy with no retryable
    categories still retries explicitly configured transient failures.
    """
    policy = config or {}
    attempts = max(1, int(policy.get("max_attempts", 1)))
    retry_on = set(policy.get("retry_on", ["timeout", "rate_limit", "transient_error", "schema_violation"]))
    initial = max(0.0, float(policy.get("initial_delay_s", 1)))
    maximum = max(0.0, float(policy.get("max_delay_s", 30)))
    jitter = float(policy.get("jitter", 0))

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt >= attempts or not (classify_error(error) & retry_on):
                raise
            delay = min(maximum, initial * (2 ** (attempt - 1)))
            if jitter:
                delay += delay * jitter * random_value()
            if on_retry:
                on_retry(attempt, error, delay)
            if delay:
                sleep(delay)
    raise RuntimeError("retry operation exited unexpectedly")


class TokenBucketRateLimiter:
    """Process-local token bucket limiter for configured tool calls."""

    def __init__(self, calls: int, per_seconds: float, clock: Callable[[], float] = time.monotonic):
        import threading

        self.capacity = max(1, int(calls))
        self.per_seconds = max(0.001, float(per_seconds))
        self.clock = clock
        self._tokens = float(self.capacity)
        self._last_refill = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self.clock()
                refill_rate = self.capacity / self.per_seconds
                self._tokens = min(self.capacity, self._tokens + (now - self._last_refill) * refill_rate)
                self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait_for = (1 - self._tokens) / refill_rate
            time.sleep(max(0.0, wait_for))


SlidingWindowRateLimiter = TokenBucketRateLimiter


@dataclass
class GraphVersion:
    """Snapshot of a graph's node definitions and healing metadata."""

    version: int
    parent: Optional[int]
    nodes: List[Dict[str, Any]]
    reason: str = "baseline"
    patch: Dict[str, Any] = field(default_factory=dict)
    trace_summary: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"
    created_at: float = field(default_factory=time.time)


class GraphRegistry:
    """Append-only version history for self-healing graph modifications."""

    def __init__(self, baseline_nodes: List[Dict[str, Any]], max_versions: Optional[int] = None):
        self.versions: List[GraphVersion] = [
            GraphVersion(version=0, parent=None, nodes=copy.deepcopy(baseline_nodes), reason="baseline", status="active")
        ]
        self.max_versions = max_versions
        self._active = 0

    @property
    def active(self) -> GraphVersion:
        return self.versions[self._active]

    def record_modification(
        self,
        nodes: List[Dict[str, Any]],
        reason: str,
        patch: Optional[Dict[str, Any]] = None,
        trace_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> GraphVersion:
        if self.max_versions is not None and len(self.versions) >= int(self.max_versions):
            raise RuntimeError(f"Maximum graph versions ({self.max_versions}) reached")

        current = self.active
        current.status = "superseded"
        next_version = GraphVersion(
            version=len(self.versions),
            parent=current.version,
            nodes=copy.deepcopy(nodes),
            reason=reason,
            patch=copy.deepcopy(patch or {}),
            trace_summary=list(trace_summary or []),
            status="active",
        )
        self.versions.append(next_version)
        self._active = next_version.version
        return next_version

    def rollback(self) -> GraphVersion:
        current = self.active
        if current.parent is None:
            raise RuntimeError("Nothing to roll back to")
        current.status = "rolled_back"
        parent = self.versions[current.parent]
        parent.status = "active"
        self._active = parent.version
        return parent


def apply_graph_patch(nodes: List[Dict[str, Any]], patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply a structured graph patch (`node_updates`, `add_nodes`, `remove_nodes`, `reroute`)."""
    updated = copy.deepcopy(nodes)
    by_id = {n.get("id"): idx for idx, n in enumerate(updated) if isinstance(n, dict) and n.get("id")}

    for upd in patch.get("node_updates", []) or []:
        if not isinstance(upd, dict) or not upd.get("id"):
            continue
        nid = upd["id"]
        if nid in by_id:
            updated[by_id[nid]] = {**updated[by_id[nid]], **copy.deepcopy(upd)}

    for new_node in patch.get("add_nodes", []) or []:
        if not isinstance(new_node, dict) or not new_node.get("id"):
            continue
        nid = new_node["id"]
        if nid in by_id:
            updated[by_id[nid]] = copy.deepcopy(new_node)
        else:
            by_id[nid] = len(updated)
            updated.append(copy.deepcopy(new_node))

    remove_ids = set(patch.get("remove_nodes", []) or [])
    if remove_ids:
        updated = [n for n in updated if n.get("id") not in remove_ids]
        by_id = {n.get("id"): idx for idx, n in enumerate(updated) if isinstance(n, dict) and n.get("id")}

    reroute = patch.get("reroute")
    if isinstance(reroute, dict) and reroute.get("from") and reroute.get("to"):
        src_id = reroute["from"]
        if src_id in by_id:
            src_node = updated[by_id[src_id]]
            rule = {"when": reroute.get("when", "true"), "to": reroute["to"]}
            routing = list(src_node.get("routing") or [])
            routing.insert(0, rule)
            src_node["routing"] = routing

    return updated


def _extract_json_patch(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class SelfHealer:
    """Configurable healer model that inspects execution traces and versions graph fixes."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a self-healing graph architect. Inspect the execution trace, failed node, "
        "and current node definitions, then return a JSON object with keys such as "
        "'reason', 'goto', 'node_updates', 'add_nodes', 'remove_nodes', and 'reroute'."
    )

    def __init__(self, config: Optional[Dict[str, Any]], registry: GraphRegistry, llm: Any = None):
        self.config = config or {}
        self.registry = registry
        self.attempt_counts: Dict[str, int] = {}

        raw_model_cfg = self.config.get("healer_model") or self.config.get("llm_config") or {}
        if isinstance(raw_model_cfg, str):
            self.healer_model_config: Dict[str, Any] = {"model": raw_model_cfg}
        else:
            self.healer_model_config = dict(raw_model_cfg)

        self.llm = llm if llm is not None else self._build_healer_llm()

    def _build_healer_llm(self) -> Any:
        if not self.healer_model_config:
            return None
        try:
            from langchain.chat_models import init_chat_model

            provider = self.healer_model_config.get("provider") or self.healer_model_config.get("model_provider", "google-genai")
            model = self.healer_model_config.get("model", "gemini-3.6-pro")
            temp = self.healer_model_config.get("temperature", 0.1)
            return init_chat_model(model=model, model_provider=provider, temperature=temp)
        except Exception:
            return None

    def extract_trace(
        self,
        state: Dict[str, Any],
        error: Optional[BaseException] = None,
        failed_node: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        raw_trace = state.get("_trace") or state.get("__trace__") or []
        for entry in raw_trace:
            trace.append(copy.deepcopy(entry) if isinstance(entry, dict) else {"event": str(entry)})
        for msg in state.get("messages", []) or []:
            trace.append({
                "type": type(msg).__name__,
                "content": getattr(msg, "content", str(msg)),
            })
        if failed_node or error is not None:
            trace.append({
                "event": "node_error",
                "node": failed_node,
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
            })
        return trace

    def propose_patch(
        self,
        trace: List[Dict[str, Any]],
        error: BaseException,
        failed_node: str,
        node_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                sys_prompt = self.healer_model_config.get("system_instruction", self.DEFAULT_SYSTEM_INSTRUCTION)
                payload = {
                    "failed_node": failed_node,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "trace": trace,
                    "current_nodes": self.registry.active.nodes,
                }
                response = self.llm.invoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=json.dumps(payload, default=str)),
                ])
                raw_content = getattr(response, "content", response)
                parsed = _extract_json_patch(raw_content)
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        error_target = ((node_info or {}).get("on_error") or {}).get("then")
        patch: Dict[str, Any] = {
            "action": "modify_graph",
            "reason": f"{type(error).__name__} in {failed_node}",
        }
        if error_target:
            patch["goto"] = error_target
            patch["reroute"] = {"from": failed_node, "to": error_target}
        return patch

    def heal(
        self,
        state: Dict[str, Any],
        error: BaseException,
        failed_node: str,
        node_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        max_attempts = int(self.config.get("max_heal_attempts_per_node", 2))
        attempts = self.attempt_counts.get(failed_node, 0)
        if attempts >= max_attempts:
            return {
                "healed": False,
                "reason": "max_attempts_exceeded",
                "version": self.registry.active.version,
            }

        self.attempt_counts[failed_node] = attempts + 1
        trace = self.extract_trace(state, error=error, failed_node=failed_node)
        patch = self.propose_patch(trace, error=error, failed_node=failed_node, node_info=node_info)

        versioning_enabled = (self.config.get("versioning") or {}).get("enabled", True)
        has_graph_mods = any(k in patch for k in ("node_updates", "add_nodes", "remove_nodes", "reroute"))
        if has_graph_mods and versioning_enabled:
            new_nodes = apply_graph_patch(self.registry.active.nodes, patch)
            if new_nodes != self.registry.active.nodes:
                self.registry.record_modification(
                    new_nodes,
                    reason=patch.get("reason", f"heal:{failed_node}"),
                    patch=patch,
                    trace_summary=trace,
                )

        goto = patch.get("goto") or ((node_info or {}).get("on_error") or {}).get("then")
        return {
            "healed": bool(goto or patch),
            "version": self.registry.active.version,
            "patch": patch,
            "goto": goto,
            "trace": trace,
            "nodes": self.registry.active.nodes,
        }

