"""
Evaluator thresholds and guarded, versioned harness optimization.
"""

from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- targets
_TARGET = re.compile(r"^(==|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$")
_OPS: Dict[str, Callable[[float, float], bool]] = {
    "==": lambda a, b: math.isclose(a, b, abs_tol=1e-9),
    "!=": lambda a, b: not math.isclose(a, b, abs_tol=1e-9),
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
}


def parse_target(target: Any) -> tuple[str, float]:
    """Parse once, fail loudly. Call at compile time so a typo can't silently mean 'never met'."""
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return "==", float(target)
    match = _TARGET.match(str(target).strip())
    if not match:
        raise ValueError(f"Invalid metric target: {target!r}")
    return match.group(1), float(match.group(2))


@dataclass(frozen=True)
class MetricCheck:
    key: str
    value: Optional[float]
    target: Any
    status: str          # "met" | "unmet" | "unknown" (no data is NOT the same as failing)
    violation: float     # distance from the target; 0 when met

    @property
    def met(self) -> bool:
        return self.status == "met"

    def __getitem__(self, key: str) -> Any:
        if key == "met":
            return self.met
        return getattr(self, key)


def target_met(value: Any, target: Any) -> bool:
    """Return True if a metric value satisfies the comparator target."""
    return check_metric("_", value, target).status == "met"


def check_metric(key: str, value: Any, target: Any) -> MetricCheck:
    op, expected = parse_target(target)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MetricCheck(key, None, target, "unknown", 0.0)
    actual = float(value)
    met = _OPS[op](actual, expected)
    return MetricCheck(key, actual, target, "met" if met else "unmet", 0.0 if met else abs(actual - expected))


def evaluate_metrics(metrics: Iterable[Dict[str, Any]], values: Dict[str, Any]) -> Dict[str, MetricCheck]:
    return {m["key"]: check_metric(m["key"], values.get(m["key"]), m.get("target"))
            for m in metrics or [] if m.get("key")}


def total_violation(checks: Dict[str, MetricCheck]) -> float:
    return sum(c.violation for c in checks.values())


# --------------------------------------------------------------------------- mutators
Mutator = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]   # (harness, ctx) -> harness
STRATEGIES: Dict[str, Mutator] = {}


def strategy(name: str) -> Callable[[Mutator], Mutator]:
    def register(fn: Mutator) -> Mutator:
        STRATEGIES[name] = fn
        return fn
    return register


def _append_once(text: str, sentence: str) -> str:
    return text if sentence in text else f"{text} {sentence}".strip()   # idempotent: no prompt bloat


@strategy("cot_prompt_injection")
def _cot(h, ctx):
    # Reasoning stays internal so schema-strict outputs (JSON) are not polluted.
    h["system_instruction"] = _append_once(
        h.get("system_instruction", ""),
        "Reason step by step internally, then output only the final answer in the required schema.")
    return h


@strategy("negative_prompting_injection")
def _negative(h, ctx):
    h["system_instruction"] = _append_once(
        h.get("system_instruction", ""),
        "Do not invent facts. If information is absent from the input, say so instead of guessing.")
    return h


@strategy("temperature_cooling")
def _cooling(h, ctx):
    if not ctx.get("capabilities", {}).get("temperature", True):
        return h                                    # model ignores temperature: don't "optimize" a no-op
    cfg = h.setdefault("llm_config", {})
    temp = cfg.get("temperature")
    if temp is not None and temp > 0:
        cooled = round(temp * 0.5, 3)
        cfg["temperature"] = 0.0 if cooled < 0.05 else cooled
    return h


@strategy("few_shot_refresh")
def _few_shot(h, ctx):
    # Signal only; the runtime re-selects examples and clears the flag.
    h.setdefault("context_policy", {}).setdefault("few_shot", {})["refresh_requested"] = True
    return h


@strategy("dynamic_prompt_template_evolution")
def _template(h, ctx):
    h["input_template"] = _append_once(
        h.get("input_template", ""), "Support every recommendation with concrete evidence from the input.")
    return h


def validate_optimization(spec: Dict[str, Any]) -> None:
    """Compile-time check: bad targets or unknown strategies fail before the first run."""
    for metric in spec.get("metrics", []) or []:
        parse_target(metric.get("target"))
    for name in (spec.get("on_underperformance", {}) or {}).get("strategies", []) or []:
        if name not in STRATEGIES:
            raise KeyError(f"Unknown optimization strategy: {name!r}")


# --------------------------------------------------------------------------- diff helpers
def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


def changed_paths(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    a, b = _flatten(old), _flatten(new)
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def _covered(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == p or path.startswith(p + ".") for p in prefixes)


# --------------------------------------------------------------------------- registry
@dataclass
class HarnessVersion:
    version: int
    parent: Optional[int]
    harness: Dict[str, Any]
    depth: int                                  # mutations since baseline/fallback
    reason: str = "baseline"
    strategies: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    parent_violation: float = 0.0               # live violation of the parent when promoted
    status: str = "candidate"   # candidate | pending_approval | active | superseded | rejected | rolled_back
    created_at: float = field(default_factory=time.time)


class HarnessRegistry:
    """Append-only version history; the agent only ever reads `.active.harness`."""

    def __init__(self, baseline: Dict[str, Any], fallback: Optional[Dict[str, Any]] = None):
        self.versions: List[HarnessVersion] = [HarnessVersion(0, None, copy.deepcopy(baseline), 0, status="active")]
        self.fallback = copy.deepcopy(fallback if fallback is not None else baseline)
        self._active = 0

    @property
    def active(self) -> HarnessVersion:
        return self.versions[self._active]

    def add(self, harness, reason, strategies=(), changed=(), status="candidate") -> HarnessVersion:
        v = HarnessVersion(len(self.versions), self.active.version, copy.deepcopy(harness),
                           self.active.depth + 1, reason, list(strategies), list(changed), status=status)
        self.versions.append(v)
        return v

    def promote(self, v: HarnessVersion, parent_violation: float = 0.0) -> None:
        self.active.status = "superseded"
        v.status, v.parent_violation, self._active = "active", parent_violation, v.version

    def rollback(self) -> HarnessVersion:
        current = self.active
        if current.parent is None:
            raise RuntimeError("Nothing to roll back to")
        current.status = "rolled_back"
        parent = self.versions[current.parent]
        parent.status, self._active = "active", parent.version
        return parent

    def reset_to_fallback(self) -> HarnessVersion:
        v = self.add(self.fallback, "fallback")
        v.depth = 0
        self.promote(v)
        return v


# --------------------------------------------------------------------------- optimizer
@dataclass
class Window:
    """Metrics aggregated over samples observed under the ACTIVE version only."""
    values: Dict[str, Any]
    sample_count: int


DEFAULT_MUTABLE = ["system_instruction", "input_template", "llm_config.temperature", "context_policy.few_shot"]


def optimize(spec: Dict[str, Any], registry: HarnessRegistry, window: Window, *,
             global_guardrails: Optional[Dict[str, Any]] = None,
             capabilities: Optional[Dict[str, Any]] = None,
             eval_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
             approval_fn: Optional[Callable[[HarnessVersion], bool]] = None) -> Dict[str, Any]:
    """One optimization tick. Returns a decision dict; caller sets agent.harness = registry.active.harness."""
    g = {**(global_guardrails or {}), **(spec.get("guardrails") or {})}
    max_depth = min(int(spec.get("max_mutation_depth", 3)), int((global_guardrails or {}).get("max_mutation_depth", 99)))
    approval_fields = set((global_guardrails or {}).get("human_approval_for", [])) | \
        set((spec.get("guardrails") or {}).get("human_approval_for", []))

    checks = evaluate_metrics(spec.get("metrics", []), window.values)
    if not checks:
        return {"changed": False, "reason": "no_metrics"}
    if window.sample_count < int(g.get("min_samples_before_mutation", 20)):
        return {"changed": False, "reason": "insufficient_samples"}
    if any(c.status == "unknown" for c in checks.values()):
        return {"changed": False, "reason": "insufficient_data", "metrics": checks}

    violation = total_violation(checks)
    active = registry.active

    # Post-promotion monitoring: worse than the version we replaced -> roll back.
    if (g.get("auto_rollback_on_regression", True) and active.parent is not None and active.reason != "fallback"
            and violation > active.parent_violation + float(g.get("regression_tolerance", 0.0))):
        parent = registry.rollback()
        return {"changed": True, "action": "rollback", "to_version": parent.version, "metrics": checks}

    if violation == 0:
        return {"changed": False, "reason": "targets_met", "metrics": checks}

    if active.depth >= max_depth:
        if active.reason == "fallback":
            return {"changed": False, "reason": "exhausted_needs_human", "metrics": checks}
        v = registry.reset_to_fallback()
        return {"changed": True, "action": "fallback", "to_version": v.version, "metrics": checks}

    action = (spec.get("on_underperformance") or {})
    mode = action.get("action")
    if mode not in ("mutate_harness", "mutate_context"):
        return {"changed": False, "reason": "no_mutation_action", "metrics": checks}

    allowed = list(g.get("mutable_fields", DEFAULT_MUTABLE))
    if mode == "mutate_context":                    # context mutations can't touch instructions/model settings
        allowed = [p for p in allowed if p.startswith(("input_template", "context_policy"))]

    proposal, applied = copy.deepcopy(active.harness), []
    for name in action.get("strategies", []):
        fn = STRATEGIES.get(name)
        if fn is None:
            continue
        proposal, applied = fn(proposal, {"capabilities": capabilities or {}}), applied + [name]

    changed = changed_paths(active.harness, proposal)
    if not changed:
        return {"changed": False, "reason": "no_op", "metrics": checks}
    forbidden = [p for p in changed if not _covered(p, allowed)]
    if forbidden:
        return {"changed": False, "reason": "forbidden_fields", "fields": forbidden}

    candidate = registry.add(proposal, f"underperformance:{','.join(sorted(k for k, c in checks.items() if c.status == 'unmet'))}",
                             applied, changed)

    # Offline gate: the candidate must beat the active version on a held-out/golden set.
    if eval_fn is not None:
        cand_checks = evaluate_metrics(spec.get("metrics", []), eval_fn(proposal))
        if any(c.status == "unknown" for c in cand_checks.values()) or total_violation(cand_checks) >= violation:
            candidate.status = "rejected"
            return {"changed": False, "reason": "no_offline_improvement", "version": candidate.version}
    elif g.get("require_offline_eval", False):
        candidate.status = "rejected"
        return {"changed": False, "reason": "offline_eval_required", "version": candidate.version}

    # Human gate: only when a field under approval was actually changed (not just declared).
    if any(_covered(p, approval_fields) for p in changed):
        if approval_fn is None or not approval_fn(candidate):
            candidate.status = "pending_approval"
            return {"changed": False, "reason": "approval_required", "version": candidate.version, "diff": changed}

    registry.promote(candidate, parent_violation=violation)
    return {"changed": True, "action": "promote", "version": candidate.version, "diff": changed, "metrics": checks}


def apply_optimization(
    agent_obj: Any,
    metrics: Dict[str, Any],
    *,
    sample_count: int = 1,
    global_guardrails: Optional[Dict[str, Any]] = None,
    approval_fn: Optional[Callable[[HarnessVersion], bool]] = None,
) -> Dict[str, Any]:
    """Run one optimization tick against an agent object using a lazily managed HarnessRegistry."""
    spec = agent_obj.get_optimization() if hasattr(agent_obj, "get_optimization") else getattr(agent_obj, "optimization", {})
    if not spec:
        return {"changed": False, "reason": "no_optimization_spec"}

    registry = getattr(agent_obj, "_harness_registry", None)
    if registry is None:
        registry = HarnessRegistry(agent_obj.harness)
        agent_obj._harness_registry = registry

    effective_guardrails = {"min_samples_before_mutation": 1, **(global_guardrails or {})}
    decision = optimize(
        spec,
        registry,
        Window(values=metrics, sample_count=sample_count),
        global_guardrails=effective_guardrails,
        approval_fn=approval_fn,
    )
    if decision.get("changed"):
        agent_obj.harness = copy.deepcopy(registry.active.harness)
        agent_obj.harness["version"] = registry.active.version
    return decision
