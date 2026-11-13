import unittest
import os
import yaml
from src.meta_agent_madcoderme.utils import deep_merge

class CompilerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = os.path.join(os.path.dirname(__file__), "jobsearch.agent.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_top_level_metadata(self):
        self.assertEqual(self.config.get("schema_version"), "1.0")
        self.assertEqual(self.config.get("entry_node"), "file_upload")
        self.assertIn("modules", self.config)
        self.assertEqual(self.config["modules"]["data_models"], "data_models")

    def test_runtime_memory_stores(self):
        runtime = self.config.get("runtime", {})
        memory_stores = runtime.get("memory", {}).get("stores", {})
        self.assertIn("short_term", memory_stores)
        self.assertIn("long_term", memory_stores)
        self.assertEqual(memory_stores["short_term"]["type"], "in_memory")
        self.assertEqual(memory_stores["long_term"]["type"], "in_memory")

    def test_harness_defaults_inheritance(self):
        defaults = self.config.get("harness_defaults", {})
        self.assertIsNotNone(defaults)

        # resume_agent overrides temperature and system_instruction, inherits provider/model
        resume_harness = self.config["agents"]["resume_agent"].get("harness", {})
        merged_resume = deep_merge(defaults, resume_harness)

        self.assertEqual(merged_resume["llm_config"]["provider"], "google-genai")
        self.assertEqual(merged_resume["llm_config"]["model"], "gemini-3.6-flash")
        self.assertEqual(merged_resume["llm_config"]["temperature"], 0.0)
        self.assertIn("Extract chronological", merged_resume["system_instruction"])
        self.assertIn("few_shot", merged_resume["context_policy"])
        self.assertIn("window_management", merged_resume["context_policy"])

        # suggestions_agent overrides model and temperature
        sugg_harness = self.config["agents"]["suggestions_agent"].get("harness", {})
        merged_sugg = deep_merge(defaults, sugg_harness)
        self.assertEqual(merged_sugg["llm_config"]["model"], "gemini-3.5-flash")
        self.assertEqual(merged_sugg["llm_config"]["temperature"], 0.7)
        self.assertEqual(merged_sugg["llm_config"]["provider"], "google-genai")

        # suggestions_editor_agent inherits llm_config completely
        editor_harness = self.config["agents"]["suggestions_editor_agent"].get("harness", {})
        merged_editor = deep_merge(defaults, editor_harness)
        self.assertEqual(merged_editor["llm_config"]["model"], "gemini-3.6-flash")
        self.assertEqual(merged_editor["context_policy"]["window_management"]["max_tokens"], 4096)

    def test_node_routing_definitions(self):
        nodes_by_id = {n["id"]: n for n in self.config.get("nodes", [])}

        # Check parallel fan-out
        resume_proc = nodes_by_id.get("node_resume_processor")
        self.assertIsNotNone(resume_proc)
        self.assertEqual(resume_proc["routing"][0]["to"], ["node_suggestions", "node_preferences"])

        # Check binary branching and default
        editor = nodes_by_id.get("node_suggestions_editor")
        self.assertIsNotNone(editor)
        self.assertEqual(editor.get("default"), "end")
        self.assertEqual(editor["routing"][0]["to"], "node_resume_updater")
        self.assertEqual(editor["routing"][1]["to"], "end")

if __name__ == "__main__":
    unittest.main()
