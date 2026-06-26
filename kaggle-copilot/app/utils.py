"""
Utility Functions Module

This module contains pure helper functions used across the application to handle 
data parsing, serialization, and formatting of raw inputs/outputs from the ADK nodes 
and Gemini models.
"""

import os
import json
import hashlib
import shutil
from typing import Any
from app.schema import KaggleState

CACHE_DIR = ".llm_cache"

def clear_llm_cache():
    """
    Clears the deterministic LLM state cache by deleting the cache directory.
    """
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        
async def run_cached_node(ctx: Any, node_func: Any, state: KaggleState) -> KaggleState:
    """
    Executes a node (typically an LLM Agent) with deterministic caching.
    
    It hashes the node's name and the input KaggleState. If a cache file exists for this hash,
    it returns the cached state instantly, bypassing the LLM API call.
    If not, it runs the node normally and saves the output to the cache.
    """
    # Create deterministic hash based on node name and input state JSON
    state_json = state.model_dump_json()
    hash_input = f"{getattr(node_func, 'name', str(node_func))}_{state_json}".encode('utf-8')
    cache_key = hashlib.md5(hash_input).hexdigest()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    # 1. Check Cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                # Ensure the loaded data is converted to KaggleState properly
                return KaggleState(**cached_data)
        except Exception as e:
            # If cache is corrupted, proceed to normal execution
            pass
            
    # 2. Execute Node
    result_obj = await ctx.run_node(node_func, state)
    result_state = to_state(result_obj)
    
    # 3. Save Cache
    try:
        with open(cache_path, "w") as f:
            f.write(result_state.model_dump_json())
    except Exception:
        pass
        
    return result_state

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
        
    data_dict = {}
    if isinstance(obj, dict):
        data_dict = obj
    elif hasattr(obj, "__dict__"):
        data_dict = obj.__dict__
        
    if data_dict:
        # Handle backward compatibility: if human_feedback is a string, convert to list
        if "human_feedback" in data_dict and isinstance(data_dict["human_feedback"], str):
            hf_str = data_dict["human_feedback"].strip()
            data_dict["human_feedback"] = [hf_str] if hf_str else []
            
        try:
            return KaggleState(**data_dict)
        except Exception:
            pass
            
    # If all conversions fail, return a fresh empty state
    return KaggleState()
