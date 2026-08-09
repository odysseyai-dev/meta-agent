# nodes.py

from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool

from langgraph.types import interrupt
from langgraph.store.base import BaseStore

class NetworkAgent():
    def __init__(self):
        self.id: str = ""
        self.type: str = ""
        self.description: str = ""
        self.input: List[str] = []
        self.output: List[str] = []
        self.routing: List[Dict[str, Any]] = []
        self.is_starting_node: bool = False
        self.trigger: str = ""
        self.tools: List[BaseTool] = []
        
        self.harness = {
            "system_prompt": "",
            "llm_config": {
                "model_provider": "google-genai",
                "model": "gemini-3.6-flash",
                "temperature": 0.5
            }
        }
    def create(self):
        self.agent = None
        
        llm_cfg = self.harness.get("llm_config", {})
        provider = llm_cfg.get("provider", "google-genai")
        model = llm_cfg.get("model", "gemini-3.6-flash")
        temp = llm_cfg.get("temperature", 0.0)
        system_instruction = self.harness.get("system_instruction") or self.harness.get("system_prompt", "")
        
        if self.type == "chat":
            self.agent = init_chat_model(
                model=model,
                model_provider=provider,
                temperature=temp
            )
        elif self.type == "react":
            self.agent = create_agent(
                model=provider + ":" + model,
                system_prompt=system_instruction,
            )
            
    def get_agent(self):
        return self.agent
    

def create_agents(agent_list: Dict[str, Any], tools: Dict[str, BaseTool]):
    agents: List[NetworkAgent] = []
    for id, agent in agent_list.items():
        new_agent = NetworkAgent()
        new_agent.id = id
        new_agent.type = agent.get("type", "chat")
        new_agent.description = agent.get("description", "")
        new_agent.input = agent.get("input", [])
        new_agent.output = agent.get("output", [])
        new_agent.routing = agent.get("routing", [])
        new_agent.is_starting_node = agent.get("is_starting_node", False)
        new_agent.trigger = agent.get("trigger_cond", None)
        new_agent.harness = agent.get("harness", {})
        new_agent.tools = [tools[tool_name] for tool_name in agent.get("tools", []) if tool_name in tools]
        
        new_agent.create()
        agents.append(new_agent)
    return agents

def create_human_node(node_info: Dict[str, Any]):
    def human_node(state: Dict[str, Any], store: BaseStore = None):
        input_type = node_info.get("input_type", "text")
        target_state_key = node_info.get("input_state", "")
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
            case "text":
                return {target_state_key: user_response}
    return human_node