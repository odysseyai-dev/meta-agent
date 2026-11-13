import unittest
from types import SimpleNamespace

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.meta_agent_madcoderme.compiler import apply_context_policy


class ContextPolicyTests(unittest.TestCase):
    def test_summarize_oldest_uses_agent_window_policy(self):
        agent = SimpleNamespace(
            harness={
                "context_policy": {
                    "window_management": {
                        "max_tokens": 30,
                        "overflow_strategy": "summarize_oldest",
                    }
                }
            }
        )
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="First user turn"),
            AIMessage(content="First assistant response"),
            HumanMessage(content="Second user turn"),
        ]

        managed_messages = apply_context_policy(messages, agent.harness.get("context_policy", {}))

        self.assertLessEqual(len(managed_messages), 3)
        self.assertEqual(managed_messages[-1].content, "Second user turn")
        self.assertTrue(any("Earlier context" in msg.content for msg in managed_messages))


if __name__ == "__main__":
    unittest.main()
