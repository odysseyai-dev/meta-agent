import importlib
import importlib.util
import os
from typing import Any, Dict, List, Optional
from jinja2 import Template
import ast
import re

def resolve_module(module_name: str, base_dirs: Optional[List[str]] = None) -> Optional[Any]:
    """
    Dynamically loads a Python module by name.
    Searches standard import path, tests.<name>, and candidate directories.
    """
    if not module_name:
        return None

    # 1. Standard import
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        pass

    # 2. Package-prefixed (e.g. tests.<name>)
    try:
        return importlib.import_module(f"tests.{module_name}")
    except (ImportError, ModuleNotFoundError):
        pass

    # 3. Search directory file paths
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "tests")]
    if base_dirs:
        search_dirs.extend(base_dirs)

    cleaned_name = module_name.split(".")[-1]
    for directory in search_dirs:
        candidate = os.path.join(directory, f"{cleaned_name}.py")
        if os.path.isfile(candidate):
            try:
                spec = importlib.util.spec_from_file_location(module_name, candidate)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
            except Exception:
                pass

    return None

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged

class ConfigResolver:
    @staticmethod
    def resolve_value(spec: Any, state: Dict[str, Any], store: Any = None) -> Any:
        """
        Resolves a value from raw YAML specs.
        Supports raw strings, Jinja2 templates, and external Python functions.
        """
        if not isinstance(spec, dict):
            return spec

        resolution_type = spec.get("type")

        if resolution_type == "template":
            template_str = spec.get("value", "")
            return Template(template_str).render(state=state)

        elif resolution_type == "function":
            module_name = spec.get("module")
            func_name = spec.get("function")
            args = spec.get("args", {})

            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)

            return func(state=state, store=store, **args)

        return spec
    
def extract_input_context(agent_obj, state: Dict[str, Any]) -> Dict[str, Any]:
    raw_input = getattr(agent_obj, "input", None)
    context = {}
    if isinstance(raw_input, dict):
        for key, ref in raw_input.items():
            if isinstance(ref, str) and ref.startswith("state."):
                state_key = ref[len("state."):]
                context[key] = state.get(state_key)
            elif isinstance(ref, str):
                context[key] = state.get(ref)
            else:
                context[key] = ref
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                state_key = item[len("state."):] if item.startswith("state.") else item
                context[state_key] = state.get(state_key)
    elif isinstance(raw_input, str):
        state_key = raw_input[len("state."):] if raw_input.startswith("state.") else raw_input
        context[state_key] = state.get(state_key)
    return context

def apply_pre_filters(context: Dict[str, Any], pre_filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Applies context_policy.pre_filters to the resolved input context before prompt rendering.

    Supported filter actions:
      - strip_keys: removes listed keys (args) from the target value if it is a dict.
        target is a dotted path like "input.resume" — the first segment is the top-level
        context key (e.g. "input"), the rest navigates into the value.

    Returns a shallow copy of `context` with filters applied.
    """
    if not pre_filters:
        return context

    result = dict(context)

    for f in pre_filters:
        action = f.get("action")
        target_path = f.get("target", "")
        args = f.get("args", [])

        if not action or not target_path:
            continue

        # Resolve target path: "input.resume" → key="input", sub_key="resume"
        parts = target_path.split(".")
        top_key = parts[0]  # e.g. "input"
        sub_path = parts[1:]  # e.g. ["resume"]

        if top_key not in result:
            continue

        if action == "strip_keys":
            if not sub_path:
                # Apply directly to the top-level value
                val = result[top_key]
                if isinstance(val, dict):
                    result[top_key] = {k: v for k, v in val.items() if k not in args}
            else:
                # Navigate one level deeper; only single-level sub-path is supported
                sub_key = sub_path[0]
                parent = result[top_key]
                if isinstance(parent, dict) and sub_key in parent:
                    target_val = parent[sub_key]
                    if isinstance(target_val, dict):
                        filtered = {k: v for k, v in target_val.items() if k not in args}
                        result[top_key] = dict(parent)
                        result[top_key][sub_key] = filtered

    return result


def render_input_prompt(agent_obj, state: Dict[str, Any]) -> str:
    input_template = agent_obj.harness.get("input_template")
    input_context = extract_input_context(agent_obj, state)
    
    if input_template:
        return Template(input_template).render(state=state, input=input_context)

    extracted_context = {k: v for k, v in input_context.items() if v}
    if not extracted_context:
        return "No specific execution context provided."
        
    formatted_blocks = []
    for key, val in extracted_context.items():
        formatted_blocks.append(f"### {key.replace('_', ' ').title()}:\n{val}")
        
    return "\n\n".join(formatted_blocks)

class StateConditionEvaluator(ast.NodeVisitor):
    def __init__(self, state: Dict[str, Any]):
        self.state = state

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if left is None or right is None:
                if isinstance(op, ast.Eq):
                    if not (left is right):
                        return False
                elif isinstance(op, ast.NotEq):
                    if not (left is not right):
                        return False
                elif isinstance(op, ast.In):
                    return False
                elif isinstance(op, ast.NotIn):
                    return True
                else:
                    return False
            else:
                if isinstance(op, ast.Eq) and not (left == right):
                    return False
                elif isinstance(op, ast.NotEq) and not (left != right):
                    return False
                elif isinstance(op, ast.Gt) and not (left > right):
                    return False
                elif isinstance(op, ast.Lt) and not (left < right):
                    return False
                elif isinstance(op, ast.GtE) and not (left >= right):
                    return False
                elif isinstance(op, ast.LtE) and not (left <= right):
                    return False
                elif isinstance(op, ast.In) and not (left in right):
                    return False
                elif isinstance(op, ast.NotIn) and not (left not in right):
                    return False
        return True

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not self.visit(value):
                    return False
            return True
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                if self.visit(value):
                    return True
            return False
        return False

    def visit_Call(self, node):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        args = [self.visit(arg) for arg in node.args]
        if func_name == "len":
            arg_val = args[0] if args else None
            if arg_val is None:
                return 0
            return len(arg_val)
        return None

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        elif isinstance(node.op, ast.USub):
            return -operand
        return operand

    def visit_Attribute(self, node):
        # Handles nested attributes like state.suggestions.is_approved
        val = self.visit(node.value)
        if isinstance(val, dict):
            return val.get(node.attr)
        return getattr(val, node.attr, None)

    def visit_Name(self, node):
        if node.id == "state":
            return self.state.get("state", self.state)
        if node.id in ("null", "None"):
            return None
        if node.id in ("true", "True"):
            return True
        if node.id in ("false", "False"):
            return False
        return self.state.get(node.id)

    def visit_Constant(self, node):
        return node.value

def evaluate_condition(condition_str: str, state: Dict[str, Any]) -> bool:
    try:
        # Pre-process YAML-style boolean/null literals for AST parsing
        clean_cond = re.sub(r'\bnull\b', 'None', condition_str)
        clean_cond = re.sub(r'\btrue\b', 'True', clean_cond)
        clean_cond = re.sub(r'\bfalse\b', 'False', clean_cond)
        tree = ast.parse(clean_cond, mode='eval')
        evaluator = StateConditionEvaluator(state)
        return bool(evaluator.visit(tree))
    except Exception:
        return False
