import os
import unittest
from unittest.mock import patch

from optimization import apply_optimization, evaluate_metrics, target_met
from resilience import retry_call
from tools import _guard_tool
from nodes import NetworkAgent


class Phase45Tests(unittest.TestCase):
    def test_retry_retries_only_configured_transient_failures(self):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("temporary timeout")
            return "ok"

        result = retry_call(
            operation,
            {"max_attempts": 3, "initial_delay_s": 0, "retry_on": ["timeout"]},
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)

    def test_retry_does_not_retry_permanent_error(self):
        calls = []

        def operation():
            calls.append(1)
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            retry_call(operation, {"max_attempts": 3, "initial_delay_s": 0, "retry_on": ["timeout"]})
        self.assertEqual(len(calls), 1)

    def test_tool_auth_is_injected_without_exposing_it_in_schema(self):
        received = {}

        def fetch(query: str, api_key: str = ""):
            received["api_key"] = api_key
            return query

        with patch.dict(os.environ, {"JOB_API_KEY": "secret"}):
            guarded = _guard_tool(fetch, {"auth": {"env": "JOB_API_KEY"}})
            self.assertEqual(guarded("python"), "python")
        self.assertEqual(received["api_key"], "secret")

    def test_tool_rate_limit_policy_is_constructed(self):
        def fetch(query: str):
            return query

        guarded = _guard_tool(fetch, {"rate_limit": {"calls": 2, "per_seconds": 60}})
        self.assertEqual(guarded("one"), "one")
        self.assertEqual(guarded("two"), "two")

    def test_metric_targets_support_yaml_comparators(self):
        self.assertTrue(target_met(0.04, "< 0.05"))
        self.assertFalse(target_met(0.10, "< 0.10"))
        result = evaluate_metrics([{"key": "error_rate", "target": "== 0"}], {"error_rate": 0})
        self.assertTrue(result["error_rate"]["met"])

    def test_optimization_respects_sample_guardrail_and_versions_mutation(self):
        agent = NetworkAgent()
        agent.harness = {"system_instruction": "Be precise.", "llm_config": {"temperature": 0.8}}
        agent.optimization = {
            "metrics": [{"key": "false_positive_rate", "target": "< 0.10"}],
            "on_underperformance": {
                "action": "mutate_harness",
                "strategies": ["negative_prompting_injection", "temperature_cooling"],
            },
            "guardrails": {"min_samples_before_mutation": 2},
        }

        blocked = apply_optimization(agent, {"false_positive_rate": 0.4}, sample_count=1)
        self.assertFalse(blocked["changed"])
        changed = apply_optimization(agent, {"false_positive_rate": 0.4}, sample_count=2)
        self.assertTrue(changed["changed"])
        self.assertEqual(agent.harness["version"], 1)
        self.assertEqual(agent.harness["llm_config"]["temperature"], 0.4)
        self.assertIn("Do not invent facts", agent.harness["system_instruction"])

    def test_system_instruction_mutation_requires_approval(self):
        agent = NetworkAgent()
        agent.harness = {"system_instruction": "Be precise."}
        agent.optimization = {
            "metrics": [{"key": "error_rate", "target": "== 0"}],
            "on_underperformance": {"action": "mutate_harness", "strategies": ["negative_prompting_injection"]},
            "guardrails": {"human_approval_for": ["system_instruction"]},
        }
        result = apply_optimization(agent, {"error_rate": 1}, sample_count=1)
        self.assertFalse(result["changed"])
        self.assertEqual(result["reason"], "approval_required")


if __name__ == "__main__":
    unittest.main()