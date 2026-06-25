"""
Utility Functions Module

This module contains pure helper functions used across the application to handle 
data parsing, serialization, and formatting of raw inputs/outputs from the ADK nodes 
and Gemini models.
"""

from typing import Any
from app.schema import KaggleState

def extract_text(node_input: Any) -> str:
    """
    Safely extracts plain text from a generic node input or ADK Content object.

    Args:
        node_input (Any): The input payload which could be a string, an ADK Content 
            object with 'parts', or any generic object.

    Returns:
        str: The extracted plain text string.
    """
    text = ""
    # Check if the input is a structured ADK Content object containing parts
    if hasattr(node_input, "parts"):
        for part in node_input.parts:
            if hasattr(part, "text"):
                text += part.text
    # If it's already a string, use it directly
    elif isinstance(node_input, str):
        text = node_input
    # Fallback to standard string representation
    else:
        text = str(node_input)
    return text

def to_state(obj: Any) -> KaggleState:
    """
    Converts a generic object, dictionary, or existing state into a standard KaggleState object.

    This ensures that regardless of what an agent or node returns (e.g. a raw dictionary 
    or a partial Pydantic model), the workflow always operates on a valid, 
    fully-formed KaggleState instance.

    Args:
        obj (Any): The object to convert. Can be a dict, KaggleState, or custom object.

    Returns:
        KaggleState: A validated KaggleState Pydantic model instance.
    """
    if isinstance(obj, KaggleState):
        return obj
    if isinstance(obj, dict):
        return KaggleState(**obj)
    if hasattr(obj, "__dict__"):
        try:
            return KaggleState(**obj.__dict__)
        except Exception:
            pass
    # If all conversions fail, return a fresh empty state
    return KaggleState()
