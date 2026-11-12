import unittest
import os
import yaml
from typing import get_origin, get_args, Annotated, List

from utils import resolve_module, deep_merge
from state import create_states, reducer_append, reducer_dedupe_by_id, get_state_scopes
from nodes import NetworkAgent, create_agents
from tools import create_tools

class Phase1VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = os.path.join(os.path.dirname(__file__), "jobsearch.agent.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    # 1. Module Resolution Tests
    def test_dynamic_module_resolution(self):
        dm_mod = resolve_module("data_models")
        self.assertIsNotNone(dm_mod)
        self.assertTrue(hasattr(dm_mod, "ResumeModel"))
        self.assertTrue(hasattr(dm_mod, "JobListingModel"))

        tools_mod = resolve_module("tool_defs")
        self.assertIsNotNone(tools_mod)
        self.assertTrue(hasattr(tools_mod, "document_parser"))
        self.assertTrue(hasattr(tools_mod, "document_editor"))

        self.assertIsNone(resolve_module("non_existent_module_xyz"))

    # 2. Reducer Unit Tests
    def test_reducer_append(self):
        self.assertEqual(reducer_append(None, ["item1"]), ["item1"])
        self.assertEqual(reducer_append([], ["item1"]), ["item1"])

        existing = ["item1"]
        result = reducer_append(existing, ["item2", "item3"])
        self.assertEqual(result, ["item1", "item2", "item3"])

        result2 = reducer_append(result, "item4")
        self.assertEqual(result2, ["item1", "item2", "item3", "item4"])

        self.assertEqual(reducer_append(result2, None), result2)

    def test_reducer_dedupe_by_id_dicts(self):
        existing = [{"id": "1", "title": "Job A"}, {"id": "2", "title": "Job B"}]
        incoming = [{"id": "2", "title": "Job B duplicate"}, {"id": "3", "title": "Job C"}]

        result = reducer_dedupe_by_id(existing, incoming)
        self.assertEqual(len(result), 3)
        self.assertEqual([x["id"] for x in result], ["1", "2", "3"])
        self.assertEqual(result[1]["title"], "Job B")

    def test_reducer_dedupe_by_id_objects(self):
        class Item:
            def __init__(self, id, val):
                self.id = id
                self.val = val

        existing = [Item("a", 10), Item("b", 20)]
        incoming = [Item("b", 99), Item("c", 30)]

        result = reducer_dedupe_by_id(existing, incoming)
        self.assertEqual(len(result), 3)
        self.assertEqual([x.id for x in result], ["a", "b", "c"])

    # 3. State Schema & Scopes Tests
    def test_create_states_with_reducers_and_scopes(self):
        state_config = self.config.get("state", {})
        modules_config = self.config.get("modules", {})
        NetworkState = create_states(state_config, modules_config=modules_config)

        annotations = NetworkState.__annotations__

        # Verify job_listings uses reducer_dedupe_by_id
        job_listings_type = annotations["job_listings"]
        self.assertEqual(get_origin(job_listings_type), Annotated)
        args = get_args(job_listings_type)
        self.assertEqual(args[1], reducer_dedupe_by_id)

        # Verify job_analysis uses reducer_append
        job_analysis_type = annotations["job_analysis"]
        self.assertEqual(get_origin(job_analysis_type), Annotated)
        args_ja = get_args(job_analysis_type)
        self.assertEqual(args_ja[1], reducer_append)

        # Verify worker scopes
        scopes = getattr(NetworkState, "__state_scopes__", {})
        self.assertEqual(scopes.get("single_job_listing"), "worker")
        self.assertEqual(scopes.get("single_job_analysis"), "worker")
        self.assertEqual(scopes.get("approved_job_analysis"), "worker")
        self.assertEqual(scopes.get("resume_file"), "global")

    # 4. Tools Creation with Dynamic Modules
    def test_create_tools_with_modules_config(self):
        tools_config = self.config.get("tools", {})
        modules_config = self.config.get("modules", {})
        tools = create_tools(tools_config, modules_config=modules_config)

        self.assertIn("document_parser", tools)
        self.assertIn("document_editor", tools)
        self.assertEqual(tools["document_parser"].name, "document_parser")

    # 5. NetworkAgent Class & YAML Design Matching
    def test_network_agent_structure_and_methods(self):
        agent = NetworkAgent()
        agent.id = "test_agent"
        agent.description = "Test Description"
        agent.input = {"resume": "state.parsed_resume", "prefs": "state.preferences"}
        agent.output = "state.test_output"
        agent.optimization = {"evaluator": "schema_validator", "max_mutation_depth": 3}
        agent.retry = {"max_attempts": 3, "backoff": "exponential"}
        agent.harness = {
            "system_instruction": "You are a test agent.",
            "llm_config": {"model": "gemini-3.6-flash", "provider": "google-genai", "temperature": 0.0},
            "fallback_llm_config": {"model": "gemini-3.5-flash", "temperature": 0.0},
            "context_policy": {"window_management": {"max_tokens": 4096}}
        }

        self.assertEqual(agent.get_output_key(), "test_output")

        test_state = {"parsed_resume": {"skills": ["Python"]}, "preferences": {"remote": True}}
        inputs = agent.resolve_inputs(test_state)
        self.assertEqual(inputs["resume"], {"skills": ["Python"]})
        self.assertEqual(inputs["prefs"], {"remote": True})

        self.assertEqual(agent.get_system_instruction(), "You are a test agent.")
        self.assertEqual(agent.get_context_policy()["window_management"]["max_tokens"], 4096)
        self.assertEqual(agent.get_retry_config()["max_attempts"], 3)
        self.assertEqual(agent.get_optimization()["evaluator"], "schema_validator")

    def test_create_agents_from_jobsearch_yaml(self):
        agents_config = self.config.get("agents", {})
        tools_config = self.config.get("tools", {})
        modules_config = self.config.get("modules", {})
        defaults = self.config.get("harness_defaults", {})

        tools = create_tools(tools_config, modules_config=modules_config)
        agents = create_agents(agents_config, tools, harness_defaults=defaults)

        agents_by_id = {a.id: a for a in agents}

        # Check resume_agent
        resume_agent = agents_by_id.get("resume_agent")
        self.assertIsNotNone(resume_agent)
        self.assertEqual(resume_agent.get_output_key(), "parsed_resume")
        self.assertIn("resume_file", resume_agent.input)
        self.assertEqual(resume_agent.harness["llm_config"]["temperature"], 0.0)
        self.assertEqual(resume_agent.harness["llm_config"]["provider"], "google-genai")
        self.assertEqual(resume_agent.optimization.get("evaluator"), "schema_validator")
        self.assertEqual(len(resume_agent.tools), 2)

        # Check application_agent retry spec
        app_agent = agents_by_id.get("application_agent")
        self.assertIsNotNone(app_agent)
        self.assertEqual(app_agent.retry.get("max_attempts"), 1)
        self.assertEqual(app_agent.get_output_key(), "applications")

if __name__ == "__main__":
    unittest.main()
