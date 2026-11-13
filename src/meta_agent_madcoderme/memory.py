# memory.py

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

def apply_context_policy(
    messages: List[BaseMessage], 
    context_policy: Optional[Dict[str, Any]] = None,
    llm: Any = None
) -> List[BaseMessage]:
    if not context_policy or not messages:
        return messages

    window_mgmt = context_policy.get("window_management", {}) or {}
    overflow_strategy = window_mgmt.get("overflow_strategy", "drop_oldest")

    if overflow_strategy == "sliding_window":
        max_messages = window_mgmt.get("max_messages", 10)
        if len(messages) <= max_messages:
            return messages
        
        system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
        retained = messages[-max_messages:]
        
        if system_msg and system_msg not in retained:
            return [system_msg] + retained
        return retained

    max_tokens = window_mgmt.get("max_tokens")
    if not max_tokens:
        return messages
    
    max_tokens = int(max_tokens)


    if overflow_strategy in ("drop_oldest", "truncate"):
        return trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True,
            allow_partial=False,
            start_on=window_mgmt.get("start_on", "human"),
        )


    elif overflow_strategy == "summarize_oldest":
        total_tokens = sum(count_tokens_approximately([m]) for m in messages)
        if total_tokens <= max_tokens:
            return messages

        system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
        chat_history = messages[1:] if system_msg else messages

        preserve_count = window_mgmt.get("preserve_recent_count", 1)
        if len(chat_history) <= preserve_count:
            return messages

        old_messages = chat_history[:-preserve_count]
        recent_messages = chat_history[-preserve_count:]

        if llm:
            custom_prompt = window_mgmt.get("summary_prompt", "Summarize the following conversation history concisely:")
            summary_res = llm.invoke([
                SystemMessage(content=custom_prompt),
                HumanMessage(content=str([m.content for m in old_messages]))
            ])
            summary_text = summary_res.content
        else:
            summary_text = "Earlier context summarized to fit window: " + " ".join(
                str(getattr(m, 'content', m)) for m in old_messages
            )

        summary_msg = AIMessage(content=f"[Context Summary]: {summary_text}")

        retained = []
        if system_msg:
            retained.append(system_msg)
        retained.append(summary_msg)
        retained.extend(recent_messages)

        trimmed = trim_messages(
            retained,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True,
            allow_partial=False,
        )
        if not any("Earlier context" in getattr(m, 'content', '') for m in trimmed):
            # Ensure summary is not discarded if trim_messages pruned it
            fallback = []
            if system_msg:
                fallback.append(system_msg)
            fallback.append(summary_msg)
            if recent_messages:
                fallback.append(recent_messages[-1])
            return fallback
        return trimmed

    return messages


def create_memory(memory_config: Optional[Dict[str, Any]] = None):
    checkpointer = None
    store = None
        
    if memory_config:
        stores = memory_config.get('stores', memory_config)
        short_term = stores.get('short_term', {})
        long_term = stores.get('long_term', {})
        
        match short_term.get('type', None):
            case "in_memory":
                checkpointer = InMemorySaver()
            case "postgres":
                checkpointer = PostgresSaver.from_conn_string(short_term.get('connection_string', ''))
                checkpointer = checkpointer.__enter__()
                checkpointer.setup()
        match long_term.get('type', None):
            case "in_memory":
                store = InMemoryStore()
            case "postgres":
                store = PostgresStore.from_conn_string(long_term.get('connection_string', ''))
                store = store.__enter__()
                store.setup()
                
    return checkpointer, store