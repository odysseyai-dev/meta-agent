import unittest
from types import SimpleNamespace
from utils import extract_input_context, render_input_prompt

class PromptRenderingTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "resume_file": "My Resume Content",
            "parsed_resume": "Parsed Experience Details",
            "suggestions": "Add more Python keywords",
            "preferences": "Remote only"
        }

    def test_extract_context_dict_format(self):
        agent = SimpleNamespace(
            input={"resume": "state.parsed_resume", "prefs": "state.preferences"},
            harness={}
        )
        context = extract_input_context(agent, self.state)
        self.assertEqual(context.get("resume"), "Parsed Experience Details")
        self.assertEqual(context.get("prefs"), "Remote only")

    def test_extract_context_legacy_list_format(self):
        agent = SimpleNamespace(
            input=["state.resume_file", "state.suggestions"],
            harness={}
        )
        context = extract_input_context(agent, self.state)
        self.assertEqual(context.get("resume_file"), "My Resume Content")
        self.assertEqual(context.get("suggestions"), "Add more Python keywords")

    def test_jinja_render_with_input_namespace(self):
        agent = SimpleNamespace(
            input={"resume": "state.parsed_resume"},
            harness={
                "input_template": "Analyze this data:\n{{ input.resume }}"
            }
        )
        rendered = render_input_prompt(agent, self.state)
        self.assertEqual(rendered, "Analyze this data:\nParsed Experience Details")

    def test_jinja_render_with_state_fallback(self):
        agent = SimpleNamespace(
            input={"resume": "state.parsed_resume"},
            harness={
                "input_template": "Resume file: {{ state.resume_file }}"
            }
        )
        rendered = render_input_prompt(agent, self.state)
        self.assertEqual(rendered, "Resume file: My Resume Content")

    def test_default_block_rendering_without_template(self):
        agent = SimpleNamespace(
            input={"parsed_resume": "state.parsed_resume"},
            harness={}
        )
        rendered = render_input_prompt(agent, self.state)
        self.assertIn("### Parsed Resume:\nParsed Experience Details", rendered)

if __name__ == "__main__":
    unittest.main()
