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
    # 1. Research phase (no structured schema, so tool works)
    prompt = f"Dataset: {node_input.dataset_characteristics}\nProblem: {node_input.problem_type_patterns}"
    if node_input.human_feedback:
        prompt += f"\nHuman Feedback / Constraints: {node_input.human_feedback}"
        
    res = await ctx.run_node(model_research_agent, node_input=prompt)
    research_results = extract_text(res)
    
    # Extract sources from research_results if any
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
    modified_state = KaggleState(**node_input.model_dump())
    modified_state.problem_type_patterns += f"\n\n--- RESEARCH STRATEGY ---\n{research_results}"
    
    # Run the agent to generate the code
    final_res = await ctx.run_node(baseline_model_agent, node_input=modified_state)
    final_state = to_state(final_res)
    
    # Keep the sources
    final_state.research_sources = sources
    
    # Clean up the injected research from the state so it doesn't clutter the next nodes
    final_state.problem_type_patterns = node_input.problem_type_patterns
    return final_state

@node(name="download_dataset_node", rerun_on_resume=True)
async def download_dataset_node(ctx: Context, node_input: KaggleState) -> KaggleState:
    # Use the python API directly to avoid LLM timeouts for large files
    res = download_kaggle_competition_data(node_input.input_text)
    ctx.state["download_status"] = res
    return node_input

# Interactive nodes decorated with @node(rerun_on_resume=False)
@node(name="ask_for_url", rerun_on_resume=False)
async def ask_for_url(ctx: Context, message: str = "Please provide a valid Kaggle competition URL to proceed."):
    yield RequestInput(message=message)

@node(name="ask_for_review", rerun_on_resume=False)
async def ask_for_review(ctx: Context, node_input: KaggleState):
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

# Deterministic File Writer Node (Only runs after user approval)
@node(name="write_code_file_node", rerun_on_resume=True)
async def write_code_file_node(ctx: Context, node_input: KaggleState) -> KaggleState:
    """Writes the approved script and report to files, and creates a Jupyter Notebook."""
    if not node_input.final_script.strip():
        return node_input
    try:
        # Write flat files
        with open("baseline_solution.py", "w") as f:
            f.write(node_input.final_script)
        with open("baseline_report.md", "w") as f:
            f.write(node_input.final_report)
            
        # Create Jupyter Notebook
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
