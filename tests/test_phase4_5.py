import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, TypedDict
import unittest
from unittest.mock import patch

from langgraph.types import Command

import builder
from graph import create_graph
from optimization import apply_optimization, evaluate_metrics, target_met
from resilience import GraphRegistry, SelfHealer, retry_call
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

    def test_self_healer_reads_trace_and_versions_graph_modification(self):
        baseline_nodes = [
            {"id": "step_a", "agent": "agent_a", "routing": [{"when": "true", "to": "end"}]},
            {"id": "recovery_node", "agent": "agent_b", "routing": [{"when": "true", "to": "end"}]},
        ]
        registry = GraphRegistry(baseline_nodes, max_versions=5)

        seen_payloads = []

        class MockHealerLLM:
            def invoke(self, messages):
                seen_payloads.append(json.loads(messages[-1].content))
                patch_body = {
                    "reason": "reroute_after_timeout",
                    "goto": "recovery_node",
                    "reroute": {"from": "step_a", "to": "recovery_node", "when": "true"},
                    "node_updates": [{"id": "step_a", "healed": True}],
                }
                return SimpleNamespace(content=f"```json\n{json.dumps(patch_body)}\n```")

        healer = SelfHealer(
            {
                "enabled": True,
                "healer_model": {"provider": "google-genai", "model": "gemini-3.6-pro", "temperature": 0.1},
                "versioning": {"enabled": True, "max_versions": 5},
                "max_heal_attempts_per_node": 2,
            },
            registry,
            llm=MockHealerLLM(),
        )

        state = {
            "__trace__": [{"node": "step_a", "status": "started"}],
            "messages": [SimpleNamespace(content="processing step_a")],
        }
        res = healer.heal(state, TimeoutError("step timed out"), "step_a", node_info=baseline_nodes[0])

        self.assertTrue(res["healed"])
        self.assertEqual(res["goto"], "recovery_node")
        self.assertEqual(res["version"], 1)
        self.assertEqual(registry.active.version, 1)
        self.assertEqual(registry.versions[0].status, "superseded")
        self.assertEqual(registry.active.status, "active")
        self.assertEqual(seen_payloads[0]["failed_node"], "step_a")
        self.assertEqual(seen_payloads[0]["trace"][0]["node"], "step_a")

        step_a_v1 = next(n for n in registry.active.nodes if n["id"] == "step_a")
        self.assertTrue(step_a_v1["healed"])
        self.assertEqual(step_a_v1["routing"][0]["to"], "recovery_node")

        rolled_back = registry.rollback()
        self.assertEqual(rolled_back.version, 0)
        self.assertEqual(registry.versions[1].status, "rolled_back")

    def test_graph_compilation_attaches_healer_and_routes_on_error(self):
        class SimpleState(TypedDict):
            messages: list
            output: Any
            _error: Any
            _heal: Any

        failing_agent = NetworkAgent()
        failing_agent.id = "failing_agent"
        failing_agent.output = "state.output"

        class FailingLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise RuntimeError("primary failure")

        failing_agent.agent = FailingLLM()

        recovery_agent = NetworkAgent()
        recovery_agent.id = "recovery_agent"
        recovery_agent.output = "state.output"

        class RecoveryLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                return SimpleNamespace(content="recovered", tool_calls=None)

        recovery_agent.agent = RecoveryLLM()

        nodes = [
            {
                "id": "primary_node",
                "agent": "failing_agent",
                "on_error": {"action": "route_to_healer", "then": "healer_fallback"},
            },
            {
                "id": "healer_fallback",
                "agent": "recovery_agent",
                "routing": [{"when": "true", "to": "end"}],
            },
        ]

        workflow = create_graph(
            SimpleState,
            [failing_agent, recovery_agent],
            nodes,
            entry_node="primary_node",
            runtime_config={
                "resilience": {"default_retry": {"max_attempts": 1}},
                "self_healing": {
                    "enabled": True,
                    "healer_model": {"provider": "google-genai", "model": "gemini-3.6-pro"},
                    "versioning": {"enabled": True, "max_versions": 5},
                },
            },
        )
        self.assertIsNotNone(workflow.__graph_registry__)
        self.assertIsNotNone(workflow.__self_healer__)

        # Stub out healer LLM to avoid external calls while still recording a graph version
        workflow.__self_healer__.llm = None
        compiled = workflow.compile()
        result = compiled.invoke({"messages": [], "output": None, "_error": None, "_heal": None})
        self.assertEqual(result["output"], "recovered")
        self.assertEqual(result["_error"]["node"], "primary_node")
        self.assertEqual(result["_heal"]["version"], 1)
        self.assertEqual(workflow.__graph_registry__.active.version, 1)

    def test_builder_generates_and_validates_agent_yaml(self):
        sample_yaml = """schema_version: "1.0"
name: Research Network
description: Searches and summarizes topics.
entry_node: research_node
state:
  topic: { type: str }
  report: { type: str }
agents:
  research_agent:
    input: { topic: state.topic }
    output: state.report
    harness:
      system_instruction: "Research thoroughly."
nodes:
  - id: research_node
    agent: research_agent
    routing:
      - { when: "true", to: end }
"""

        class MockBuilderLLM:
            def invoke(self, messages):
                return SimpleNamespace(content=f"```yaml\n{sample_yaml}```")

        yaml_text = builder.generate_agent_yaml(
            {"name": "Research Network", "description": "Build a research network"},
            llm=MockBuilderLLM(),
        )
        parsed = builder.validate_agent_yaml(yaml_text)
        self.assertEqual(parsed["entry_node"], "research_node")
        self.assertEqual(builder.ensure_agent_yaml_path("my_net.yaml"), Path("my_net.agent.yaml"))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_target = Path(tmpdir) / "research_net"
            saved_path = builder.main(
                ["Build a research network", "--name", "Research Network", "-o", str(out_target)],
                llm=MockBuilderLLM(),
            )
            self.assertEqual(saved_path.name, "research_net.agent.yaml")
            self.assertTrue(saved_path.exists())
            self.assertIn("research_node", saved_path.read_text(encoding="utf-8"))

    def test_builder_generates_companion_modules_ready_to_compile(self):
        import sys
        import yaml
        from state import create_states
        from tools import create_tools
        from nodes import create_agents

        sample_yaml = """schema_version: "1.0"
name: Support Triage
description: Triages tickets and replies.
entry_node: intake_node
modules:
  data_models: custom_models
  tools: custom_tools
state:
  ticket: { type: TicketModel }
  reply: { type: str }
tools:
  ticket_lookup:
    impl: custom_tools.ticket_lookup
    description: Looks up a customer ticket.
    input_schema: { ticket_id: str }
    auth: { type: api_key, env: TICKET_API_KEY }
agents:
  triage_agent:
    tools: [ticket_lookup]
    input: { ticket: state.ticket }
    output: state.reply
    harness:
      system_instruction: "Triage ticket."
nodes:
  - id: intake_node
    agent: triage_agent
    routing:
      - { when: "true", to: end }
"""
        module_json = json.dumps({
            "custom_models.py": (
                "from pydantic import BaseModel\n\n"
                "class TicketModel(BaseModel):\n"
                "    id: str\n"
                "    subject: str = ''\n"
            ),
            "custom_tools.py": (
                "def ticket_lookup(ticket_id: str, api_key: str = ''):\n"
                "    return {'id': ticket_id, 'status': 'open'}\n"
            ),
        })

        class DualMockLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(content=f"```yaml\n{sample_yaml}```")
                return SimpleNamespace(content=f"```json\n{module_json}```")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "support_triage.agent.yaml"
            saved = builder.main(
                ["Support triage workflow", "-o", str(out_file)],
                llm=DualMockLLM(),
            )
            models_path = Path(tmpdir) / "custom_models.py"
            tools_path = Path(tmpdir) / "custom_tools.py"
            self.assertTrue(saved.exists())
            self.assertTrue(models_path.exists())
            self.assertTrue(tools_path.exists())
            self.assertIn("class TicketModel", models_path.read_text(encoding="utf-8"))
            self.assertIn("def ticket_lookup", tools_path.read_text(encoding="utf-8"))

            cfg = yaml.safe_load(saved.read_text(encoding="utf-8"))
            sys.path.insert(0, tmpdir)
            try:
                net_state = create_states(cfg["state"], modules_config=cfg["modules"])
                loaded_tools = create_tools(cfg["tools"], modules_config=cfg["modules"])
                loaded_agents = create_agents(cfg["agents"], loaded_tools)
                wf = create_graph(net_state, loaded_agents, cfg["nodes"], tools=loaded_tools, entry_node=cfg["entry_node"])
                compiled = wf.compile()
                self.assertIn("intake_node", compiled.get_graph().nodes)
                self.assertIn("ticket_lookup", loaded_tools)
            finally:
                sys.path.remove(tmpdir)


if __name__ == "__main__":
    unittest.main()