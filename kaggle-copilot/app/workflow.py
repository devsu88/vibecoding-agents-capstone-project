"""
Workflow Orchestrator Module

This module defines the primary state machine that coordinates the execution of all 
agents and deterministic nodes. It handles the interactive lifecycle of the Kaggle Copilot, 
including pausing for user inputs (RequestInput) and handling resume logic.
"""

from typing import Any
from dotenv import load_dotenv

from google.adk import Workflow, Event, Context
from google.adk.events import RequestInput
from google.adk.workflow import node
from google.adk.apps import App, ResumabilityConfig

from app.schema import KaggleState
from app.utils import extract_text, to_state
from app.tools import extract_and_validate_kaggle_url
from app.agents import (
    competition_ingestion_node,
    problem_understanding_node,
    eda_node,
    preprocessing_node,
    feature_engineering_node,
    evaluation_node,
    report_generator_node,
    code_critic_node
)
from app.nodes import (
    download_dataset_node,
    baseline_model_node,
    write_code_file_node
)

# Load environment variables (e.g. GEMINI_API_KEY)
load_dotenv()

@node(name="kaggle_copilot_workflow", rerun_on_resume=True)
async def kaggle_copilot_workflow(ctx: Context, node_input: Any) -> str:
    """
    The main state machine orchestrator for the Kaggle Copilot.
    
    This workflow uses `ctx.state` to keep track of exactly which step of the 
    pipeline the user is currently in. If the app stops or restarts, ADK's 
    resumability features will jump back into this state machine with the saved state.
    
    Args:
        ctx (Context): The ADK Context holding workflow variables and state.
        node_input (Any): The user's chat input or the output from a previous node.
        
    Yields:
        RequestInput: To pause execution and ask the user for feedback or URL.
        Event: To output final system messages back to the UI.
    """
    # 1. State Initialization
    if "kaggle_state" in ctx.state:
        state = KaggleState(**ctx.state["kaggle_state"])
    else:
        state = KaggleState(input_text="")

    if "step" not in ctx.state:
        ctx.state["step"] = "ask_url"
        
    # 2. Main Event Loop
    while True:
        # Phase 1: Await and Validate URL
        if ctx.state["step"] == "ask_url":
            text = extract_text(node_input)
            url, is_valid = extract_and_validate_kaggle_url(text)
            
            if is_valid and url:
                state.input_text = url
                ctx.state["step"] = "process_url"
                ctx.state["ask_url_yielded"] = False
                ctx.state["kaggle_state"] = state.model_dump()
            else:
                ctx.state["ask_url_yielded"] = True
                yield RequestInput(interrupt_id="url_input", message="Please provide a valid Kaggle competition URL to proceed.")
                return

        # Phase 2: Metadata extraction
        elif ctx.state["step"] == "process_url":
            if not ctx.state.get("metadata_extracted", False):
                state = to_state(await ctx.run_node(competition_ingestion_node, state))
                ctx.state["metadata_extracted"] = True
            
            # Transition to the interactive download prompt
            ctx.state["step"] = "ask_download"
            ctx.state["kaggle_state"] = state.model_dump()
            
        # Phase 2.5: Interactive Dataset Download Decision
        elif ctx.state["step"] == "ask_download":
            if not ctx.state.get("ask_download_yielded", False):
                ctx.state["ask_download_yielded"] = True
                yield RequestInput(interrupt_id="ask_download", message="Do you want me to download the dataset locally in this workspace so you can test the code immediately? (Reply 'yes' or 'no')")
                return
            else:
                feedback = extract_text(node_input).strip().lower()
                
                # If the user says NO, skip download
                if feedback in ["no", "n", "skip"]:
                    ctx.state["ask_download_yielded"] = False # reset
                    ctx.state["dataset_downloaded"] = False
                    
                    # Proceed to modeling
                    state = to_state(await ctx.run_node(problem_understanding_node, state))
                    state = to_state(await ctx.run_node(eda_node, state))
                    ctx.state["step"] = "modeling_loop"
                    ctx.state["kaggle_state"] = state.model_dump()
                    
                # If the user says YES (or anything else like "fatto" during a retry)
                else:
                    ctx.state["ask_download_yielded"] = False # reset
                    
                    if not ctx.state.get("dataset_downloaded", False):
                        state = to_state(await ctx.run_node(download_dataset_node, state))
                        res = ctx.state.get("download_status", "")
                        
                        if "403" in res or "Forbidden" in res:
                            ctx.state["ask_download_yielded"] = True # keep yielded to loop back here
                            yield RequestInput(message="⚠️ **Action Required:** The Kaggle API returned a 403 (Forbidden) error. This happens because you haven't accepted the competition rules yet.\n\nPlease go to the 'Rules' page of the competition on Kaggle, click **'I Understand and Accept'**, and reply 'yes' here to retry the download (or 'no' to skip the download and continue).")
                            return
                        elif "Error" in res:
                            ctx.state["ask_download_yielded"] = True
                            yield RequestInput(message=f"⚠️ **Download Error:** {res}\n\nPlease resolve the issue and reply 'yes' to retry (or 'no' to skip the download and continue).")
                            return
                            
                        ctx.state["dataset_downloaded"] = True

                    # Move to the modeling loop
                    state = to_state(await ctx.run_node(problem_understanding_node, state))
                    state = to_state(await ctx.run_node(eda_node, state))
                    ctx.state["step"] = "modeling_loop"
                    ctx.state["kaggle_state"] = state.model_dump()

        # Phase 3: The Modeling Loop (preprocessing, feature engineering, modeling, review)
        elif ctx.state["step"] == "modeling_loop":
            state = to_state(await ctx.run_node(preprocessing_node, state))
            state = to_state(await ctx.run_node(feature_engineering_node, state))
            state = to_state(await ctx.run_node(baseline_model_node, state))
            state = to_state(await ctx.run_node(evaluation_node, state))
            state = to_state(await ctx.run_node(report_generator_node, state))
            state = to_state(await ctx.run_node(code_critic_node, state))

            ctx.state["step"] = "ask_review"
            ctx.state["kaggle_state"] = state.model_dump()

        # Phase 4: User Review and Approval
        elif ctx.state["step"] == "ask_review":
            review_idx = ctx.state.get("review_count", 0)
            interrupt_id = f"review_input_{review_idx}"
            
            if not ctx.state.get("ask_review_yielded", False):
                # FIRST TIME: Yield RequestInput to show the generated plan and ask for approval
                ctx.state["ask_review_yielded"] = True
                
                critic_note = ""
                if state.critic_feedback:
                    critic_note = f"\n\n🕵️‍♂️ **Critic's Note:** {state.critic_feedback}\n"
                    
                sources_note = ""
                if state.research_sources:
                    sources_note = f"\n\n🔗 **Research Sources:**\n{state.research_sources}\n"
                
                message = (
                    f"Problem Type: {state.problem_type_patterns}\n"
                    f"Preprocessing: {state.preprocessing_strategy}\n"
                    f"Models: {state.baseline_strategies}"
                    f"{critic_note}"
                    f"{sources_note}\n\n"
                    "Do you approve this plan and code? (Reply 'approve' to save to file, or provide feedback for revision)"
                )
                yield RequestInput(interrupt_id=interrupt_id, message=message)
                return
            else:
                # RESUMING: Check the user's feedback
                feedback = extract_text(node_input)
                ctx.state["ask_review_yielded"] = False # reset for next time
                
                if "approve" in feedback.lower() or "yes" in feedback.lower() or feedback.strip() == "":
                    ctx.state["step"] = "finalize"
                else:
                    # User requested changes, loop back to the modeling loop with human_feedback injected
                    state.human_feedback = feedback
                    ctx.state["step"] = "modeling_loop"
                    ctx.state["review_count"] = review_idx + 1
                    ctx.state["kaggle_state"] = state.model_dump()

        # Phase 5: Writing Output Artifacts
        elif ctx.state["step"] == "finalize":
            state = to_state(await ctx.run_node(write_code_file_node, state))
            
            ctx.state["step"] = "completed"
            ctx.state["kaggle_state"] = state.model_dump()
            
            summary_message = (
                "🎉 **Workflow Completed Successfully!**\n\n"
                "Your baseline deliverables have been generated and saved to your workspace:\n"
                "- 📄 **Python Script:** `baseline_solution.py`\n"
                "- 📝 **Markdown Report:** `baseline_report.md`\n"
                "- 📓 **Jupyter Notebook:** `baseline_solution.ipynb`\n\n"
                "You can open them directly from your editor. To iterate or refine the results further, simply type your request in the chat!"
            )
            # Emit final message to UI
            yield Event(output={
                "message": summary_message,
                "final_script": state.final_script,
                "final_report": state.final_report,
                "final_notebook": state.final_notebook
            })
            return

        # Phase 6: Continuous Refinement Loop
        elif ctx.state["step"] == "completed":
            # If the user sends a message after the workflow has finished, treat it as refinement feedback
            feedback = extract_text(node_input)
            if feedback.strip():
                state.human_feedback = feedback
                ctx.state["step"] = "modeling_loop"
                ctx.state["kaggle_state"] = state.model_dump()
                continue
            else:
                return

# Setup the root agent DAG and expose the App
root_agent = Workflow(
    name="kaggle_copilot_workflow",
    edges=[("START", kaggle_copilot_workflow)]
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True)
)
