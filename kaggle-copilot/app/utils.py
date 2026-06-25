from typing import Any
from app.schema import KaggleState

def extract_text(node_input: Any) -> str:
    text = ""
    if hasattr(node_input, "parts"):
        for part in node_input.parts:
            if hasattr(part, "text"):
                text += part.text
    elif isinstance(node_input, str):
        text = node_input
    else:
        text = str(node_input)
    return text

def to_state(obj: Any) -> KaggleState:
    if isinstance(obj, KaggleState):
        return obj
    if isinstance(obj, dict):
        return KaggleState(**obj)
    if hasattr(obj, "__dict__"):
        try:
            return KaggleState(**obj.__dict__)
        except Exception:
            pass
    return KaggleState()
