"""Evaluator thresholds and guarded, versioned harness mutations."""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, Iterable, Optional


_TARGET = re.compile(r"^(==|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$")


def target_met(value: Any, target: Any) -> bool:
    if not isinstance(target, str):
        return value == target
    match = _TARGET.match(target.strip())
    if not match:
        return False
    operator, raw_expected = match.groups()
    expected = float(raw_expected)
    actual = float(value)
    return {"==": actual == expected, "!=": actual != expected, ">=": actual >= expected,
            "<=": actual <= expected, ">": actual > expected, "<": actual < expected}[operator]


def evaluate_metrics(metrics: Iterable[Dict[str, Any]], values: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    for metric in metrics or []:
        key = metric.get("key")
        if key:
            results[key] = {"value": values.get(key), "target": metric.get("target"),
                            "met": key in values and target_met(values[key], metric.get("target"))}
    return results


def mutate_harness(harness: Dict[str, Any], strategies: Iterable[str], *, version: int = 0) -> Dict[str, Any]:
    mutated = copy.deepcopy(harness)
    instruction = mutated.get("system_instruction", "")
    for strategy in strategies or []:
        if strategy == "cot_prompt_injection" and "step by step" not in instruction.lower():
            mutated["system_instruction"] = (instruction + " Explain your reasoning step by step before the final output.").strip()
        elif strategy == "negative_prompting_injection":
            mutated["system_instruction"] = (instruction + " Do not invent facts; reject unsupported claims.").strip()
        elif strategy == "temperature_cooling":
            config = mutated.setdefault("llm_config", {})
            config["temperature"] = max(0.0, float(config.get("temperature", 0.2)) * 0.5)
        elif strategy == "few_shot_refresh":
            mutated.setdefault("context_policy", {}).setdefault("few_shot", {})["refresh"] = True
        elif strategy == "dynamic_prompt_template_evolution":
            mutated["input_template"] = (mutated.get("input_template", "") + "\nUse concrete evidence for every recommendation.").strip()
    mutated["version"] = version + 1
    return mutated


def apply_optimization(agent: Any, metric_values: Dict[str, Any], sample_count: int = 0,
                       approval_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
                       global_guardrails: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = agent.get_optimization() if hasattr(agent, "get_optimization") else getattr(agent, "optimization", {})
    guardrails = dict(global_guardrails or {})
    guardrails.update(spec.get("guardrails", {}) or {})
    if sample_count < int(guardrails.get("min_samples_before_mutation", 0)):
        return {"changed": False, "reason": "insufficient_samples"}
    checks = evaluate_metrics(spec.get("metrics", []), metric_values)
    if checks and all(item["met"] for item in checks.values()):
        return {"changed": False, "reason": "targets_met", "metrics": checks}
    action = spec.get("on_underperformance", {}).get("action")
    if action not in ("mutate_harness", "mutate_context"):
        return {"changed": False, "reason": "no_mutation_action", "metrics": checks}
    current = getattr(agent, "harness", {})
    next_version = int(current.get("version", 0))
    proposal = mutate_harness(current, spec.get("on_underperformance", {}).get("strategies", []), version=next_version)
    requires_approval = "system_instruction" in spec.get("guardrails", {}).get("human_approval_for", [])
    if requires_approval and (approval_callback is None or not approval_callback(proposal)):
        return {"changed": False, "reason": "approval_required", "proposal": proposal, "metrics": checks}
    agent.harness = proposal
    return {"changed": True, "harness": proposal, "metrics": checks}
