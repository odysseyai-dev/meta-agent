# compiler.py

import os
import argparse
from pathlib import Path
import yaml

from langgraph.types import Send, interrupt, Command
from langgraph.graph.message import add_messages

from graph import create_graph
from memory import create_memory, apply_context_policy
from nodes import create_agents
from state import create_states
from tools import create_tools

def print_result(result):
    from pprint import pprint

    print("=" * 80)
    print("PARSED LANGCHAIN / GEMINI EXECUTION TRACE")
    print("=" * 80)

    for i, message in enumerate(result.get("messages", []), 1):
        msg_type = type(message).__name__
        print(f"\n[Message {i}] Type: {msg_type}")
        print("-" * 40)

        # Print main message text content
        content = getattr(message, "content", None)
        if content:
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        print(f"Content:\n{block['text']}")
                    else:
                        pprint(block)
            else:
                print(f"Content:\n{content}")
        else:
            print("Content: <EMPTY>")

        # Print tool call details if present
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("\nTool Calls:")
            pprint(tool_calls)

        # Print tool call ID for ToolMessage
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            print(f"Tool Call ID Response: {tool_call_id}")

        # Print usage metadata (token counts) if available
        usage = getattr(message, "usage_metadata", None)
        if usage:
            print(f"Token Usage: {usage}")

    # Print remaining key-value pairs in the Result dictionary
    print("\n" + "=" * 80)
    print("OTHER RESULT KEYS")
    print("=" * 80)
    for key, value in result.items():
        if key != "messages":
            print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Meta-Agent Compiler")
    parser.add_argument("file", default=None, help="Path to the agent YAML configuration file", nargs="?")
    args = parser.parse_args()

    agent_config = None
    if args.file:
        if not args.file.endswith(".agent.yaml"):
            raise ValueError("The configuration file must have a .agent.yaml extension.")
        
        yaml_path = Path(args.file)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as file:
        try:
            agent_config = yaml.safe_load(file)
        except yaml.YAMLError as error:
            print(f"Error parsing YAML file: {error}")

    modules = agent_config.get('modules', {})
    network_nodes = agent_config.get('nodes', [])
    memory_config = agent_config.get('runtime', {}).get('memory', None) or agent_config.get('memory_config', None)
    entry_node = agent_config.get('entry_node', None)
    harness_defaults = agent_config.get('harness_defaults', {})
    config = {"configurable": {"thread_id": "session-1"}}
    
    network_state = create_states(agent_config.get('state', {}), modules)
    tools = create_tools(agent_config.get('tools', {}), modules)
    agents = create_agents(agent_config.get('agents', {}), tools, harness_defaults=harness_defaults)
    checkpointer, store = create_memory(memory_config)

    workflow = create_graph(
        network_state,
        agents,
        network_nodes,
        memory_config=memory_config,
        tools=tools,
        entry_node=entry_node,
        runtime_config=agent_config.get('runtime', {}),
        evaluator_config=agent_config.get('evaluators', {}),
        modules_config=modules,
    )
    
    compiled = workflow.compile(checkpointer=checkpointer, store=store)
    compiled.get_graph().print_ascii()
    
    default_state = { key: [] if key == "messages" else None for key in network_state.__annotations__.keys() }
    default_state.update({"resume_file": "Abrar's Resume"})
    result = compiled.invoke(default_state, config)
    
    while "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value
        print(f"\n[INTERRUPT DETECTED] Prompt: {interrupt_info.get('prompt')}")
        user_input = input(">> Enter response: ")
        result = compiled.invoke(Command(resume=user_input), config=config)
    
    print_result(result)
    
if __name__ == "__main__":
    main()
    