# tools.py

from functools import wraps
import inspect
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.tools import tool, BaseTool

from utils import resolve_module
from resilience import TokenBucketRateLimiter, retry_call


def _guard_tool(tool_func, tool_info: Dict[str, Any]):
    rate_config = tool_info.get("rate_limit") or {}
    limiter = None
    if rate_config.get("calls") is not None and rate_config.get("per_seconds") is not None:
        limiter = TokenBucketRateLimiter(rate_config["calls"], rate_config["per_seconds"])

    auth_config = tool_info.get("auth") or {}
    env_name = auth_config.get("env")
    credential = os.environ.get(env_name) if env_name else None
    signature = inspect.signature(tool_func)

    @wraps(tool_func)
    def guarded(*args, **kwargs):
        if limiter:
            limiter.acquire()
        call_kwargs = dict(kwargs)
        if credential:
            parameters = signature.parameters
            if any(parameter in parameters for parameter in ("api_key", "token", "credential", "auth")):
                for parameter in ("api_key", "token", "credential", "auth"):
                    if parameter in parameters and parameter not in call_kwargs:
                        call_kwargs[parameter] = credential
                        break
            elif any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
                call_kwargs[env_name.lower()] = credential
        return retry_call(lambda: tool_func(*args, **call_kwargs), tool_info.get("retry"))

    return guarded

def create_tools(tool_list: Dict[str, Any], modules_config: Optional[Dict[str, str]] = None) -> Dict[str, BaseTool]:
    modules_config = modules_config or {}
    dm_module_name = modules_config.get("data_models", "data_models")
    data_models = resolve_module(dm_module_name)

    tools_module_name = modules_config.get("tools", "tool_defs")
    tool_defs = resolve_module(tools_module_name)

    tools = {}
    for tool_name, tool_info in tool_list.items():
        impl_name = tool_info.get("impl", "")
        if impl_name and "." in impl_name:
            mod_part, func_part = impl_name.rsplit(".", 1)
            target_mod = resolve_module(mod_part) or tool_defs
            tool_func = getattr(target_mod, func_part, None) if target_mod else None
        else:
            func_name = impl_name or tool_name
            tool_func = getattr(tool_defs, func_name, None) if tool_defs else None

        if not tool_func and tool_defs:
            tool_func = getattr(tool_defs, tool_name, None)

        if tool_func:
            raw_schema = tool_info.get("input_schema") or tool_info.get("input", "")
            schema = getattr(data_models, raw_schema, None) if (isinstance(raw_schema, str) and data_models) else None
            tool_kwargs = {
                "name_or_callable": tool_name,
                "description": tool_info.get("description", "")
            }
            if schema and isinstance(schema, type) and issubclass(schema, BaseModel):
                tool_kwargs["args_schema"] = schema

            tools[tool_name] = tool(**tool_kwargs)(_guard_tool(tool_func, tool_info))
    return tools
