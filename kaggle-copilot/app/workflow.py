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

load_dotenv()

# Main Dynamic Orchestrator Workflow
@node(name="kaggle_copilot_workflow", rerun_on_resume=True)
async def kaggle_copilot_workflow(ctx: Context, node_input: Any) -> str:
    if "kaggle_state" in ctx.state:
        state = KaggleState(**ctx.state["kaggle_state"])
    else:
        state = KaggleState(input_text="")

    if "step" not in ctx.state:
        ctx.state["step"] = "ask_url"
        
    while True:
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

        elif ctx.state["step"] == "process_url":
            if not ctx.state.get("metadata_extracted", False):
                state = to_state(await ctx.run_node(competition_ingestion_node, state))
                ctx.state["metadata_extracted"] = True
            
            if not ctx.state.get("dataset_downloaded", False):
                state = to_state(await ctx.run_node(download_dataset_node, state))
                res = ctx.state.get("download_status", "")
                if "403" in res or "Forbidden" in res:
                    yield RequestInput(message="⚠️ **Azione Richiesta:** L'API di Kaggle ha restituito un errore 403 (Forbidden). Questo accade perché non hai ancora accettato il regolamento della competizione.\n\nVai alla pagina 'Rules' della competizione su Kaggle, clicca su **'I Understand and Accept'**, e scrivi 'fatto' qui per riprovare il download.")
                    return
                elif "Error" in res:
                    yield RequestInput(message=f"⚠️ **Errore di download:** {res}\n\nRisolvi il problema e scrivi 'riprova' per continuare.")
                    return
                ctx.state["dataset_downloaded"] = True

            state = to_state(await ctx.run_node(problem_understanding_node, state))
            state = to_state(await ctx.run_node(eda_node, state))
            ctx.state["step"] = "modeling_loop"
            ctx.state["kaggle_state"] = state.model_dump()

        elif ctx.state["step"] == "modeling_loop":
            state = to_state(await ctx.run_node(preprocessing_node, state))
            state = to_state(await ctx.run_node(feature_engineering_node, state))
            state = to_state(await ctx.run_node(baseline_model_node, state))
            state = to_state(await ctx.run_node(evaluation_node, state))
            state = to_state(await ctx.run_node(report_generator_node, state))
            state = to_state(await ctx.run_node(code_critic_node, state))

            ctx.state["step"] = "ask_review"
            ctx.state["kaggle_state"] = state.model_dump()

        elif ctx.state["step"] == "ask_review":
            review_idx = ctx.state.get("review_count", 0)
            interrupt_id = f"review_input_{review_idx}"
            
            if not ctx.state.get("ask_review_yielded", False):
                # FIRST TIME: Yield RequestInput
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
                # RESUMING: Check node_input
                feedback = extract_text(node_input)
                ctx.state["ask_review_yielded"] = False # reset for next time
                
                if "approve" in feedback.lower() or "yes" in feedback.lower() or feedback.strip() == "":
                    ctx.state["step"] = "finalize"
                else:
                    state.human_feedback = feedback
                    ctx.state["step"] = "modeling_loop"
                    ctx.state["review_count"] = review_idx + 1
                    ctx.state["kaggle_state"] = state.model_dump()

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
            yield Event(output={
                "message": summary_message,
                "final_script": state.final_script,
                "final_report": state.final_report,
                "final_notebook": state.final_notebook
            })
            return

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

# Setup the root agent and expose the App
root_agent = Workflow(
    name="kaggle_copilot_workflow",
    edges=[("START", kaggle_copilot_workflow)]
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True)
)
