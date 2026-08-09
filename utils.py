import importlib
from typing import Any, Dict, Any
from jinja2 import Template
import ast

from langgraph.graph import END

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
    
def render_input_prompt(agent_obj, state: Dict[str, Any]) -> str:
    input_template = agent_obj.harness.get("input_template")
    
    if input_template:
        return Template(input_template).render(state=state)

    input_keys = agent_obj.input if isinstance(agent_obj.input, list) else [agent_obj.input]
    extracted_context = {k: state.get(k) for k in input_keys if k in state and state.get(k)}
    
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
            if isinstance(op, ast.Eq) and not (left == right):
                return False
            elif isinstance(op, ast.NotEq) and not (left != right):
                return False
            elif isinstance(op, ast.Gt) and not (left > right):
                return False
            elif isinstance(op, ast.Lt) and not (left < right):
                return False
        return True

    def visit_Attribute(self, node):
        # Handles nested attributes like state.suggestions.is_approved
        val = self.visit(node.value)
        if isinstance(val, dict):
            return val.get(node.attr)
        return getattr(val, node.attr, None)

    def visit_Name(self, node):
        if node.id == "state":
            return self.state
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
        clean_cond = condition_str.replace(" null", " None").replace(" true", " True").replace(" false", " False")
        tree = ast.parse(clean_cond, mode='eval')
        evaluator = StateConditionEvaluator(state)
        return bool(evaluator.visit(tree))
    except Exception:
        return False