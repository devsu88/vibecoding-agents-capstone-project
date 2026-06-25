"""
Deterministic Nodes Module

This module defines deterministic, custom Python nodes that execute complex logic 
outside of the standard LLM generation loop. These nodes are orchestrated by ADK 
and can perform I/O bound tasks, UI interruptions (RequestInput), or multi-agent chaining.
"""

from google.adk.workflow import node
from google.adk import Context
from google.adk.events import RequestInput
from app.schema import KaggleState
from app.agents import model_research_agent, baseline_model_agent
from app.tools import download_kaggle_competition_data
from app.utils import extract_text, to_state
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

@node(name="baseline_model_node", rerun_on_resume=True)
async def baseline_model_node(ctx: Context, node_input: KaggleState) -> KaggleState:
    """
    Orchestrates the research and generation of the baseline model.
    
    This node first calls the `model_research_agent` to browse the web for the best
    algorithms, and then passes those findings to `baseline_model_agent` to generate
    the actual Python code.
    
    Args:
        ctx (Context): The ADK execution context.
        node_input (KaggleState): The current workflow state.
        
    Returns:
        KaggleState: The updated state containing the baseline code and research sources.
    """
    # 1. Research phase (no structured schema, so custom_web_search tool works freely)
    prompt = f"Dataset: {node_input.dataset_characteristics}\nProblem: {node_input.problem_type_patterns}"
    if node_input.human_feedback:
        prompt += f"\nHuman Feedback / Constraints: {node_input.human_feedback}"
        
    res = await ctx.run_node(model_research_agent, node_input=prompt)
    research_results = extract_text(res)
    
    # Extract sources from research_results to clean up the prompt
    sources = ""
    if "SOURCES:" in research_results:
        parts = research_results.split("SOURCES:")
        research_results = parts[0].strip()
        sources = parts[1].strip()
    elif "Sources:" in research_results:
        parts = research_results.split("Sources:")
        research_results = parts[0].strip()
        sources = parts[1].strip()
    
    # 2. Generation phase (structured output schema)
    # Inject research results temporarily so the baseline agent can read them
    modified_state = KaggleState(**node_input.model_dump())
    modified_state.problem_type_patterns += f"\n\n--- RESEARCH STRATEGY ---\n{research_results}"
    
    # Run the agent to generate the scikit-learn/gradient boosting code
    final_res = await ctx.run_node(baseline_model_agent, node_input=modified_state)
    final_state = to_state(final_res)
    
    # Store the parsed research sources
    final_state.research_sources = sources
    
    # Clean up the injected research from the state so it doesn't clutter the next nodes
    final_state.problem_type_patterns = node_input.problem_type_patterns
    return final_state

import asyncio

@node(name="download_dataset_node", rerun_on_resume=True)
async def download_dataset_node(ctx: Context, node_input: KaggleState) -> KaggleState:
    """
    Executes the dataset download using the Kaggle API.
    
    We use asyncio.to_thread to run the synchronous Kaggle API call in a background 
    thread. This prevents the download of large datasets from blocking the main 
    asyncio event loop (which would freeze the Streamlit UI).
    
    Args:
        ctx (Context): The ADK execution context.
        node_input (KaggleState): The current workflow state.
        
    Returns:
        KaggleState: The identical state, with the download status string stored in ctx.state.
    """
    res = await asyncio.to_thread(download_kaggle_competition_data, node_input.input_text)
    # Save the status in the context so the orchestrator can inspect it for errors (e.g. 403)
    ctx.state["download_status"] = res
    return node_input

# ---------------------------------------------------------------------------
# Interactive Nodes
# ---------------------------------------------------------------------------

@node(name="ask_for_url", rerun_on_resume=False)
async def ask_for_url(ctx: Context, message: str = "Please provide a valid Kaggle competition URL to proceed."):
    """Yields a RequestInput to pause execution and ask the user for a Kaggle URL."""
    yield RequestInput(message=message)

@node(name="ask_for_review", rerun_on_resume=False)
async def ask_for_review(ctx: Context, node_input: KaggleState):
    """
    Formats the generated plan and yields a RequestInput to ask the user for approval.
    
    Displays the problem type, preprocessing strategy, baseline strategy, code critic 
    notes, and research sources in the chat UI.
    """
    critic_note = ""
    if node_input.critic_feedback:
        critic_note = f"\n\n🕵️‍♂️ **Critic's Note:** {node_input.critic_feedback}\n"
        
    sources_note = ""
    if node_input.research_sources:
        sources_note = f"\n\n🔗 **Research Sources:**\n{node_input.research_sources}\n"
        
    message = (
        f"Problem Type: {node_input.problem_type_patterns}\n"
        f"Preprocessing: {node_input.preprocessing_strategy}\n"
        f"Models: {node_input.baseline_strategies}"
        f"{critic_note}"
        f"{sources_note}\n\n"
        "Do you approve this plan and code? (Reply 'approve' to save to file, or provide feedback for revision)"
    )
    yield RequestInput(message=message)

# ---------------------------------------------------------------------------
# File I/O Nodes
# ---------------------------------------------------------------------------

@node(name="write_code_file_node", rerun_on_resume=True)
async def write_code_file_node(ctx: Context, node_input: KaggleState) -> KaggleState:
    """
    Writes the final approved script and markdown report to the local filesystem.
    It also compiles a ready-to-run Jupyter Notebook containing the report and the code.
    
    Args:
        ctx (Context): The ADK execution context.
        node_input (KaggleState): The fully populated workflow state.
        
    Returns:
        KaggleState: The state updated with the raw notebook JSON string.
    """
    if not node_input.final_script.strip():
        return node_input
        
    try:
        # Write flat text files
        with open("baseline_solution.py", "w") as f:
            f.write(node_input.final_script)
        with open("baseline_report.md", "w") as f:
            f.write(node_input.final_report)
            
        # Create and write Jupyter Notebook
        nb = new_notebook()
        nb.cells.append(new_markdown_cell(node_input.final_report))
        nb.cells.append(new_code_cell(node_input.final_script))
        
        nb_content = nbformat.writes(nb)
        with open("baseline_solution.ipynb", "w") as f:
            f.write(nb_content)
            
        node_input.final_notebook = nb_content
    except Exception as e:
        print(f"Failed to write files: {e}")
        
    return node_input
