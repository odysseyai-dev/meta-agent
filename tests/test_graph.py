import unittest
import os
import yaml
from types import SimpleNamespace

from src.meta_agent_madcoderme.state import create_states
from src.meta_agent_madcoderme.tools import create_tools
from src.meta_agent_madcoderme.nodes import create_agents, NetworkAgent
from src.meta_agent_madcoderme.memory import create_memory
from src.meta_agent_madcoderme.graph import (
    create_graph, 
    make_node_executor, 
    make_swarm_executor,
    make_swarm_fanout,
    make_swarm_worker,
    make_consumer_executor, 
    make_review_node
)

class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = os.path.join(os.path.dirname(__file__), "jobsearch.agent.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_full_graph_compilation(self):
        modules_config = self.config.get("modules", {})
        network_state = create_states(self.config.get("state", {}), modules_config=modules_config)
        tools = create_tools(self.config.get("tools", {}), modules_config=modules_config)
        agents = create_agents(self.config.get("agents", {}), tools, harness_defaults=self.config.get("harness_defaults", {}))
        memory_config = self.config.get("runtime", {}).get("memory", None)
        checkpointer, store = create_memory(memory_config)

        workflow = create_graph(
            network_state,
            agents,
            self.config.get("nodes", []),
            memory_config=memory_config,
            tools=tools,
            entry_node=self.config.get("entry_node")
        )

        compiled = workflow.compile(checkpointer=checkpointer, store=store)
        graph_nodes = set(compiled.get_graph().nodes.keys())

        expected_nodes = {
            "__start__",
            "file_upload",
            "node_preferences",
            "node_resume_processor",
            "node_suggestions",
            "node_suggestions_editor",
            "node_resume_updater",
            "node_job_search",
            "node_job_analyzer",
            "node_job_analyzer_worker",   # Send-based swarm worker node
            "node_application_review",
            "node_application",
            "__end__"
        }
        self.assertTrue(expected_nodes.issubset(graph_nodes))

    def test_swarm_fanout_emits_send_objects(self):
        """make_swarm_fanout should return one Send per unfiltered item."""
        from langgraph.types import Send

        agent = NetworkAgent()
        agent.id = "mock_fanout"
        agent.output = "state.single_job_analysis"

        node_info = {
            "id": "node_job_analyzer",
            "type": "swarm",
            "iterator": {
                "loop_over": "state.job_listings",
                "yield_to": "state.single_job_listing",
                "filter": "item.active == true"
            },
            "collect": {
                "from": "state.single_job_analysis",
                "to": "state.job_analysis",
                "reducer": "append"
            }
        }

        fanout_fn = make_swarm_fanout(agent, node_info)
        test_state = {
            "job_listings": [
                {"id": "job-1", "active": True},
                {"id": "job-2", "active": False},   # filtered out
                {"id": "job-3", "active": True},
            ]
        }

        sends = fanout_fn(test_state)
        self.assertEqual(len(sends), 2)
        self.assertTrue(all(isinstance(s, Send) for s in sends))
        # Each Send targets the worker node
        self.assertEqual(sends[0].node, "node_job_analyzer_worker")
        self.assertEqual(sends[1].node, "node_job_analyzer_worker")
        # Worker state has item injected under yield_to key
        self.assertEqual(sends[0].arg["single_job_listing"]["id"], "job-1")
        self.assertEqual(sends[1].arg["single_job_listing"]["id"], "job-3")

    def test_swarm_executor_iteration_and_collect(self):
        # Mock agent that analyzes a single job
        agent = NetworkAgent()
        agent.id = "mock_analyzer"
        agent.output = "state.single_job_analysis"

        # Mock agent.get_agent to return an object whose invoke returns a mock analysis
        class MockLLM:
            def bind_tools(self, tools): return self
            def invoke(self, messages):
                return SimpleNamespace(content={"fit_score": 85}, tool_calls=None)

        agent.agent = MockLLM()

        node_info = {
            "id": "node_job_analyzer",
            "type": "swarm",
            "iterator": {
                "loop_over": "state.job_listings",
                "yield_to": "state.single_job_listing"
            },
            "collect": {
                "from": "state.single_job_analysis",
                "to": "state.job_analysis",
                "reducer": "append"
            }
        }

        executor = make_swarm_executor(agent, node_info)
        test_state = {
            "job_listings": [
                {"id": "job-1", "title": "AI Engineer"},
                {"id": "job-2", "title": "Data Scientist"}
            ]
        }

        result = executor(test_state)
        self.assertIn("job_analysis", result)
        self.assertEqual(len(result["job_analysis"]), 2)
        self.assertEqual(result["job_analysis"][0], {"fit_score": 85})

    def test_review_node_auto_approval(self):
        node_info = {
            "id": "node_application_review",
            "type": "human_in_the_loop",
            "review": {
                "source": "state.job_analysis",
                "pre_filter": "item.score >= state.preferences.min_fit_score",
                "select_to": "state.approved_job_analysis"
            },
            "auto_approve_if": "state.preferences.auto_apply == true and item.score >= 90"
        }

        review_fn = make_review_node(node_info)
        test_state = {
            "preferences": {"min_fit_score": 80, "auto_apply": True},
            "job_analysis": [
                {"id": "job-1", "score": 95},  # pre-filter: pass, auto-approve: pass
                {"id": "job-2", "score": 70},  # pre-filter: fail
            ]
        }

        result = review_fn(test_state)
        self.assertIn("approved_job_analysis", result)
        self.assertEqual(len(result["approved_job_analysis"]), 1)
        self.assertEqual(result["approved_job_analysis"][0]["id"], "job-1")

    def test_consumer_executor(self):
        agent = NetworkAgent()
        agent.id = "mock_consumer"
        agent.output = "state.applications"

        class MockLLM:
            def bind_tools(self, tools): return self
            def invoke(self, messages):
                return SimpleNamespace(content="application_submitted", tool_calls=None)

        agent.agent = MockLLM()

        node_info = {
            "id": "node_application",
            "type": "consumer",
            "iterator": {
                "loop_over": "state.approved_job_analysis",
                "yield_to": "state.approved_job_analysis"
            }
        }

        consumer_fn = make_consumer_executor(agent, node_info)
        test_state = {
            "approved_job_analysis": [{"id": "job-1"}, {"id": "job-2"}]
        }

        result = consumer_fn(test_state)
        self.assertIn("applications", result)
        self.assertEqual(len(result["applications"]), 2)
        self.assertEqual(result["applications"], ["application_submitted", "application_submitted"])

if __name__ == "__main__":
    unittest.main()
