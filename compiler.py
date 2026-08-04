import yaml
import operator
import inspect
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass

from typing_extensions import Required
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.messages.utils import count_tokens_approximately
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langmem.short_term import SummarizationNode, RunningSummary

import data_models

class NetworkAgent():
    def __init__(self):
        self.id: str = ""
        self.type: str = ""
        self.description: str = ""
        self.input: List[str] = []
        self.output: List[str] = []
        self.routing: List[Dict[str, Any]] = []
        self.is_starting_node: bool = False
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
        
        if self.type == "chat":
            self.agent = init_chat_model(
                model=model,
                model_provider=provider,
                temperature=temp
            )
        elif self.type == "react":
            self.agent = create_agent(
                model=provider + ":" + model,
                system_prompt=self.harness['system_prompt'],
            )
            
    def get_agent(self):
        return self.agent

def create_states(StateList: Dict[str, str]):
    states = {}
    states["messages"] = Annotated[List[BaseMessage], add_messages]
    def get_type(type: str):
        if type.startswith("array[") and type.endswith("]"):
            inner_type = type[6:-1]
            return List[get_type(inner_type)]
        else:
            return getattr(data_models, type, Any)
    
    for state, type in StateList.items():
        states[state] = get_type(type)
        
    return TypedDict("NetworkState", states)

def create_tools(tool_list: Dict[str, Any]) -> Dict[str, BaseTool]:
    tools = {}
    import tool_defs
    for tool_name, tool_info in tool_list.items():
        tool_func = getattr(tool_defs, tool_name, None)
        if tool_func:
            schema = getattr(data_models, tool_info.get('input', ''), None)
            tool_kwargs = {
                "name": tool_name,
                "description": tool_info.get('description', '')
            }
            if schema and isinstance(schema, type) and issubclass(schema, BaseModel):
                tool_kwargs["args_schema"] = schema

            tools[tool_name] = tool(**tool_kwargs)(tool_func)
    return tools
   
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
        new_agent.harness = agent.get("harness", {})
        new_agent.tools = [tools[tool_name] for tool_name in agent.get("tools", []) if tool_name in tools]
        
        new_agent.create()
        agents.append(new_agent)
    return agents

def demo(state):
    print("States", state)

def make_node_executor(agent_obj):
    def node_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        llm = agent_obj.get_agent()
        
        input_keys = agent_obj.input if isinstance(agent_obj.input, list) else [agent_obj.input]
        context_val = {k: state.get(k) for k in input_keys if k in state}

        long_term_memories = []
        if store:
            user_namespace = ("memories", "global")
            mem_items = store.search(user_namespace)
            long_term_memories = [item.value for item in mem_items]

        sys_prompt = agent_obj.harness.get("system_prompt", "")
        if long_term_memories:
            sys_prompt += f"\n\nLong-term Context/Memories: {long_term_memories}"
        
        messages = [SystemMessage(content=sys_prompt)]
        
        existing_messages = state.get("messages", [])
        if existing_messages:
            messages.extend(existing_messages)
            
        current_user_msg = HumanMessage(content=str(context_val))
        messages.append(current_user_msg)
        
        # Bind tools
        bound_llm = llm.bind_tools(agent_obj.tools) if agent_obj.tools else llm
        
        # Track new messages generated during this node's execution
        node_new_messages = [current_user_msg]
        
        response = bound_llm.invoke(messages)
        messages.append(response)
        node_new_messages.append(response)

        # Handle tool call loop
        while response.tool_calls:
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                t_id = tool_call["id"]
                
                # Find matching tool object
                matching_tool = next((t for t in agent_obj.tools if t.name == t_name), None)
                if matching_tool:
                    try:
                        tool_result = matching_tool.invoke(t_args)
                    except Exception as e:
                        tool_result = f"Error executing tool {t_name}: {str(e)}"
                else:
                    tool_result = f"Error: Tool {t_name} is not registered on this agent."
                
                # Create ToolMessage and append
                tool_msg = ToolMessage(content=str(tool_result), tool_call_id=t_id)
                messages.append(tool_msg)
                node_new_messages.append(tool_msg)
            
            # Re-invoke LLM with tool responses
            response = bound_llm.invoke(messages)
            messages.append(response)
            node_new_messages.append(response)

        out_key = agent_obj.output[0] if isinstance(agent_obj.output, list) and agent_obj.output else "output"
        return {
            out_key: response.content,
            "messages": node_new_messages
        }
        
    return node_fn

def create_graph(
    network_state: TypedDict, 
    agent_list: List[NetworkAgent], 
    network_nodes: List[Dict[str, Any]], 
    memory_config: Optional[Dict[str, Any]] = None,
    tools: Optional[Dict[str, BaseTool]] = None):
    
    workflow = StateGraph(state_schema=network_state)
    
    has_summarizer = memory_config and memory_config.get("short_term", {}).get("context_strategy") == "summarize"
    if has_summarizer:
        summarizer = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=agent_list[0].get_agent() if agent_list else None,
            max_tokens=256,
            max_tokens_before_summary=256,
            max_summary_tokens=128)
        workflow.add_node("context_summarizer", summarizer)
        workflow.set_entry_point("context_summarizer")
        
    for n in network_nodes:
        if not n.get('agent', None):
            print('Adding node ', n['id'])
            workflow.add_node(n['id'], demo)
        else:
            print('Adding node ', n['id'], ' with agent ', agent_list[[a.id for a in agent_list].index(n['agent'])])
            workflow.add_node(n['id'], make_node_executor(agent_list[[a.id for a in agent_list].index(n['agent'])]))
            
        if n.get('is_starting_node', False):
            if has_summarizer:
                workflow.add_edge("context_summarizer", n['id'])
            else:
                workflow.set_entry_point(n['id'])
        
        if n.get("to", None):
            for target_node in n['to']:
                print('Adding edge from ', n['id'], ' to ', target_node)
                workflow.add_edge(n['id'], target_node) 
        if n.get('routing', None):
            def create_router(rules):
                def router(state: Dict[str, Any]) -> str:
                    for rule in rules:
                        eval_globals = {"state": state, "null": None, "true": True, "false": False, "__builtins__": {}}
                        import re
                        clean_cond = re.sub(r'state\.([a-zA-Z0-9_]+)', r'state.get("\1")', rule['condition'])
                        if eval(clean_cond, eval_globals):
                            return rule['to']
                    return END
                return router
            
            path_map = {}
            for rule in n['routing']:
                target = rule.get('to')
                dest = target if target is not None else END
                path_map[dest] = dest
            path_map[END] = END
            
            workflow.add_conditional_edges(n['id'], create_router(n['routing']), path_map)
            
        
            
    return workflow

def main():
    agent_config = None
    
    with open('jobsearch.agent.yaml', 'r') as file:
        try:
            agent_config = yaml.safe_load(file)
        except yaml.YAMLError as error:
            print(f"Error parsing YAML file: {error}")
    
    network_state = create_states(agent_config.get('state', {}))
    tools = create_tools(agent_config.get('tools', {}))
    agents = create_agents(agent_config.get('agents', {}), tools)
    network_nodes = agent_config.get('nodes', [])
    memory_config = agent_config.get('memory_config', None)
    
    checkpointer = None
    store = None
    
    if memory_config:
        short_term = memory_config.get('short_term', {})
        long_term = memory_config.get('long_term', {})
        
        match short_term.get('type', None):
            case "in_memory":
                checkpointer = InMemorySaver()
            case "postgres":
                checkpointer = PostgresSaver.from_conn_string(short_term.get('connection_string', ''))
        match long_term.get('type', None):
            case "in_memory":
                store = InMemoryStore()
            case "postgres":
                store = PostgresStore.from_conn_string(long_term.get('connection_string', ''))
               
    
    workflow = create_graph(network_state, agents, network_nodes, memory_config=memory_config, tools=tools)
    
    compiled = workflow.compile(checkpointer=checkpointer, store=store)
    compiled.get_graph().print_ascii()
    
    default_state = { key: None for key in network_state.__annotations__.keys() }
    default_state.update({"resume_file": "Abrar's Resume"})
    result = compiled.invoke(default_state, {"configurable": {"thread_id": "1"}})
    print("Result", result)
    
if __name__ == "__main__":
    main()
    