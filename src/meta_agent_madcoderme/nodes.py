# nodes.py

from typing import Any, Dict, List, Optional
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool

from langgraph.types import interrupt
from langgraph.store.base import BaseStore
from .utils import deep_merge, extract_input_context, ConfigResolver

class NetworkAgent:
    """Represents an agent in the agentic network with harness, retry, and optimization specs."""
    def __init__(self):
        self.id: str = ""
        self.description: str = ""
        self.type: str = "chat"
        self.input: Any = {}
        self.output: Any = ""
        self.tools: List[BaseTool] = []

        self.harness: Dict[str, Any] = {
            "system_instruction": "",
            "llm_config": {
                "provider": "google-genai",
                "model": "gemini-3.6-flash",
                "temperature": 0.2
            },
            "fallback_llm_config": {},
            "output_validation": {},
            "context_policy": {}
        }
        self.optimization: Dict[str, Any] = {}
        self.retry: Dict[str, Any] = {}

        self.agent: Any = None
        self.fallback_agent: Any = None

    def create(self):
        """Initializes primary and fallback chat/react agents according to harness configuration."""
        self.agent = self._build_model(self.harness.get("llm_config", {}))
        
        fallback_cfg = self.harness.get("fallback_llm_config", {})
        if fallback_cfg:
            # Fallback configuration inherits primary provider unless specified
            merged_fallback = dict(self.harness.get("llm_config", {}))
            merged_fallback.update(fallback_cfg)
            self.fallback_agent = self._build_model(merged_fallback)

    def _build_model(self, llm_cfg: Dict[str, Any]) -> Any:
        provider = llm_cfg.get("provider") or llm_cfg.get("model_provider", "google-genai")
        model = llm_cfg.get("model", "gemini-3.6-flash")
        temp = llm_cfg.get("temperature", 0.0)
        system_instruction = self.harness.get("system_instruction") or self.harness.get("system_prompt", "")

        try:
            if self.type == "chat":
                return init_chat_model(
                    model=model,
                    model_provider=provider,
                    temperature=temp
                )
            elif self.type == "react":
                return create_agent(
                    model=f"{provider}:{model}",
                    system_prompt=system_instruction,
                )
        except Exception:
            return None
        return None

    def get_agent(self) -> Any:
        return self.agent

    def get_fallback_agent(self) -> Any:
        return self.fallback_agent or self.agent

    def get_output_key(self) -> str:
        """Returns the normalized state field name to write to (stripping 'state.')."""
        raw_output = self.output
        if isinstance(raw_output, list) and raw_output:
            out_key = raw_output[0]
        elif isinstance(raw_output, str) and raw_output:
            out_key = raw_output
        else:
            out_key = "output"

        if isinstance(out_key, str) and out_key.startswith("state."):
            out_key = out_key[len("state."):]
        return out_key

    def resolve_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves input mappings against the runtime state."""
        return extract_input_context(self, state)

    def get_system_instruction(self, state: Optional[Dict[str, Any]] = None, store: Optional[BaseStore] = None) -> str:
        raw_sys_spec = self.harness.get("system_instruction") or self.harness.get("system_prompt", "")
        return ConfigResolver.resolve_value(raw_sys_spec, state=state or {}, store=store)

    def get_context_policy(self) -> Dict[str, Any]:
        return self.harness.get("context_policy", {})

    def get_retry_config(self) -> Dict[str, Any]:
        return self.retry

    def get_optimization(self) -> Dict[str, Any]:
        return self.optimization

def create_agents(
    agent_list: Dict[str, Any],
    tools: Dict[str, BaseTool],
    harness_defaults: Optional[Dict[str, Any]] = None
) -> List[NetworkAgent]:
    agents: List[NetworkAgent] = []
    harness_defaults = harness_defaults or {}

    for agent_id, agent_cfg in agent_list.items():
        new_agent = NetworkAgent()
        new_agent.id = agent_id
        new_agent.description = agent_cfg.get("description", "")
        new_agent.type = agent_cfg.get("type", "chat")
        new_agent.input = agent_cfg.get("input", {})
        new_agent.output = agent_cfg.get("output", "")
        new_agent.optimization = agent_cfg.get("optimization", {})
        new_agent.retry = agent_cfg.get("retry", {})

        agent_harness = agent_cfg.get("harness", {})
        new_agent.harness = deep_merge(harness_defaults, agent_harness)

        tool_names = agent_cfg.get("tools", [])
        new_agent.tools = [tools[t] for t in tool_names if t in tools]

        new_agent.create()
        agents.append(new_agent)

    return agents

def create_human_node(node_info: Dict[str, Any]):
    def human_node(state: Dict[str, Any], store: BaseStore = None):
        input_type = node_info.get("input_type", "text")
        target_state_key = node_info.get("input_state", "")
        if not target_state_key and "review" in node_info:
            target_state_key = node_info["review"].get("select_to", "")
        if isinstance(target_state_key, str) and target_state_key.startswith("state."):
            target_state_key = target_state_key[len("state."):]

        user_response = interrupt({
            "node_id": node_info.get("id"),
            "input_type": input_type,
            "prompt": node_info.get("input_prompt", f"Please provide input for {target_state_key}")
        })
        match input_type:
            case "file":
                with open(user_response, 'r', encoding='utf-8') as file:
                    content = file.read()
                return {target_state_key: content}
            case "text" | "form" | _:
                return {target_state_key: user_response}
    return human_node
