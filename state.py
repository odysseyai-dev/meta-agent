# state.py

from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from utils import resolve_module

def reducer_append(existing: Optional[List[Any]], new: Any) -> List[Any]:
    """Appends new item(s) to an existing list state."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    if isinstance(new, list):
        return existing + new
    return existing + [new]

def reducer_dedupe_by_id(existing: Optional[List[Any]], new: Any) -> List[Any]:
    """Merges new item(s) into existing list state without duplicate IDs."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    items_to_add = new if isinstance(new, list) else [new]

    def extract_id(item: Any) -> Any:
        if isinstance(item, dict):
            return item.get("id")
        return getattr(item, "id", None)

    existing_ids = {extract_id(item) for item in existing if extract_id(item) is not None}
    result = list(existing)
    for item in items_to_add:
        item_id = extract_id(item)
        if item_id is None or item_id not in existing_ids:
            if item_id is not None:
                existing_ids.add(item_id)
            result.append(item)
    return result

BUILTIN_REDUCERS = {
    "append": reducer_append,
    "dedupe_by_id": reducer_dedupe_by_id,
}

def get_state_scopes(state_list: Dict[str, Any]) -> Dict[str, str]:
    """Extracts scope definition for each state property (default: global)."""
    scopes = {}
    for state_name, type_info in state_list.items():
        if isinstance(type_info, dict) and "scope" in type_info:
            scopes[state_name] = type_info["scope"]
        else:
            scopes[state_name] = "global"
    return scopes

def create_states(state_list: Dict[str, Any], modules_config: Optional[Dict[str, str]] = None):
    modules_config = modules_config or {}
    dm_module_name = modules_config.get("data_models", "data_models")
    data_models = resolve_module(dm_module_name)

    hooks_module_name = modules_config.get("hooks", "hooks")
    hooks_module = resolve_module(hooks_module_name)

    states = {}
    states["messages"] = Annotated[List[BaseMessage], add_messages]

    def get_type(type_info: Any):
        if isinstance(type_info, dict):
            type_str = type_info.get("type", "Any")
        else:
            type_str = str(type_info) if type_info is not None else "Any"

        if isinstance(type_str, str) and type_str.startswith("array[") and type_str.endswith("]"):
            inner_type = type_str[6:-1]
            return List[get_type(inner_type)]
        else:
            return getattr(data_models, type_str, Any) if data_models else Any

    for state_name, type_info in state_list.items():
        base_t = get_type(type_info)
        reducer_fn = None
        if isinstance(type_info, dict) and "reducer" in type_info:
            r_name = type_info["reducer"]
            reducer_fn = BUILTIN_REDUCERS.get(r_name)
            if not reducer_fn and hooks_module:
                reducer_fn = getattr(hooks_module, r_name, None)

        if reducer_fn:
            states[state_name] = Annotated[base_t, reducer_fn]
        else:
            states[state_name] = base_t

    network_state = TypedDict("NetworkState", states)
    network_state.__state_scopes__ = get_state_scopes(state_list)
    return network_state
