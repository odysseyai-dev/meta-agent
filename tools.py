# tools.py

from typing import Any, Dict, List, Optional, TypedDict, Annotated
from pydantic import BaseModel

from langchain_core.tools import tool, BaseTool

import os
import importlib.util

file_path = os.path.join(os.getcwd(), "data_models.py")

spec = importlib.util.spec_from_file_location("data_models", file_path)
data_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_models)

def create_tools(tool_list: Dict[str, Any]) -> Dict[str, BaseTool]:
    tools = {}
    import tests.tool_defs as tool_defs
    for tool_name, tool_info in tool_list.items():
        tool_func = getattr(tool_defs, tool_name, None)
        if tool_func:
            schema = getattr(data_models, tool_info.get('input', ''), None)
            tool_kwargs = {
                "name_or_callable": tool_name,
                "description": tool_info.get('description', '')
            }
            if schema and isinstance(schema, type) and issubclass(schema, BaseModel):
                tool_kwargs["args_schema"] = schema

            tools[tool_name] = tool(**tool_kwargs)(tool_func)
    return tools