"""Built-in evaluator functions and evaluator dispatch."""

from __future__ import annotations

from typing import Any, Dict, Optional


def schema_validator(output: Any, state: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
    valid = output is not None
    return {"schema_valid": valid, "json_parse_errors": 0 if valid else 1}


def grounding_checker(output: Any, state: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
    """Provide a conservative grounding signal for structured node output."""
    source = state or {}
    text = str(output)
    grounded = not text or any(str(value) in text for value in source.values() if value is not None)
    return {"grounded": grounded, "field_hallucination_rate": 0.0 if grounded else 1.0}


def user_feedback(output: Any = None, feedback: Any = None, **_: Any) -> Dict[str, Any]:
    if isinstance(feedback, dict) and "accepted" in feedback and "total" in feedback:
        total = max(1, int(feedback["total"]))
        return {"suggestion_acceptance_rate": float(feedback["accepted"]) / total}
    return {"suggestion_acceptance_rate": 0.0}
