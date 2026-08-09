# graph.py

from typing import List, Dict, Any, Optional, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore

from memory import apply_context_policy
from nodes import NetworkAgent, create_human_node
from utils import ConfigResolver, render_input_prompt, evaluate_condition

def make_node_executor(agent_obj):
    def node_fn(state: Dict[str, Any], store: BaseStore = None) -> Dict[str, Any]:
        trigger = agent_obj.trigger
        if trigger:
            if not evaluate_condition(trigger, state):
                print(f"[SKIP] Node {agent_obj['id']} trigger condition '{trigger}' not met.")
                return {}
            
        llm = agent_obj.get_agent()

        long_term_memories = []
        if store:
            user_namespace = ("memories", "global")
            mem_items = store.search(user_namespace)
            long_term_memories = [item.value for item in mem_items]

        raw_sys_spec = agent_obj.harness.get("system_instruction") or agent_obj.harness.get("system_prompt", "")
        resolved_sys_prompt = ConfigResolver.resolve_value(raw_sys_spec, state=state, store=store)
        
        if long_term_memories:
            resolved_sys_prompt += f"\n\nLong-term Context/Memories: {long_term_memories}"
        
        messages = [SystemMessage(content=resolved_sys_prompt)]
        
        existing_messages = state.get("messages") or []
        if existing_messages:
            messages.extend(existing_messages)
        
        transformer_spec = agent_obj.harness.get("input_transformer")
        if transformer_spec:
            user_prompt_text = ConfigResolver.resolve_value(transformer_spec, state=state, store=store)
        else:
            user_prompt_text = render_input_prompt(agent_obj, state)
            
        current_user_msg = HumanMessage(content=user_prompt_text)
        messages.append(current_user_msg)
        messages = apply_context_policy(messages, agent_obj.harness.get("context_policy", {}), llm)
        
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
    
    for n in network_nodes:
        if not n.get('agent', None):
            print('Adding node ', n['id'])
            node_type = n.get("type", "")
            if node_type == "human_in_the_loop":
                workflow.add_node(n['id'], create_human_node(n))
        else:
            print('Adding node ', n['id'], ' with agent ', agent_list[[a.id for a in agent_list].index(n['agent'])])
            workflow.add_node(n['id'], make_node_executor(agent_list[[a.id for a in agent_list].index(n['agent'])]))
            
        if n.get('is_starting_node', False):
            workflow.set_entry_point(n['id'])
        
        if n.get("to", None):
            for target_node in n['to']:
                print('Adding edge from ', n['id'], ' to ', target_node)
                workflow.add_edge(n['id'], target_node) 
        if n.get('routing', None):
            def create_router(rules):
                def router(state: Dict[str, Any], store: BaseStore = None) -> str:
                    for rule in rules:
                        cond = rule.get("condition")
                        # Dictionary-based resolver rules
                        if isinstance(cond, dict):
                            res = ConfigResolver.resolve_value(cond, state=state, store=store)
                            if bool(res):
                                return rule['to']
                                
                        # String condition evaluation (AST safe parser)
                        elif isinstance(cond, str):
                            if evaluate_condition(cond, state):
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