# state.py

from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

import os
import importlib.util

file_path = os.path.join(os.getcwd(), "data_models.py")

spec = importlib.util.spec_from_file_location("data_models", file_path)
data_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_models)

def create_states(state_list: Dict[str, str]):
    states = {}
    states["messages"] = Annotated[List[BaseMessage], add_messages]
    def get_type(type: str):
        if type.startswith("array[") and type.endswith("]"):
            inner_type = type[6:-1]
            return List[get_type(inner_type)]
        else:
            return getattr(data_models, type, Any)
    
    for state, type in state_list.items():
        states[state] = get_type(type)
        
    return TypedDict("NetworkState", states)