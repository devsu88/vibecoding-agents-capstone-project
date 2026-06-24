# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import json
import socket
import hashlib
import ipaddress
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from dotenv import load_dotenv
from typing import Any
from pydantic import BaseModel

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk import Workflow, Event, Context
from google.adk.events import RequestInput
from google.adk.workflow import node

load_dotenv()

KB_FILE = "knowledge_base.json"

class KaggleState(BaseModel):
    input_text: str = ""
    dataset_characteristics: str = ""
    problem_type_patterns: str = ""
    preprocessing_strategy: str = ""
    preprocessing_code: str = ""
    baseline_strategies: str = ""
    baseline_code: str = ""
    evaluation_code: str = ""
    final_script: str = ""
    human_feedback: str = ""
    past_mistakes: str = ""
    model_research_notes: str = ""
    verification_passed: bool = False
    verification_error: str = ""
    final_report: str = ""

def record_sandbox_error(error_msg: str, script: str):
    """Saves a unique error to the knowledge base to prevent future recurrence."""
    kb = []
    if os.path.exists(KB_FILE):
        try:
            with open(KB_FILE, "r") as f:
                kb = json.load(f)
        except Exception:
            pass
            
    # Hash the core error to avoid duplicates
    lines = error_msg.strip().split('\n')
    core_error = "\n".join(lines[-3:]) if len(lines) > 3 else error_msg
    error_hash = hashlib.md5(core_error.encode()).hexdigest()
    
    if any(entry.get("hash") == error_hash for entry in kb):
        return
        
    entry = {
        "hash": error_hash,
        "error": core_error,
        "full_traceback": error_msg[-1000:], 
        "context_script": script[:500] + "..."
    }
    kb.append(entry)
    
    # Keep only the latest 15 errors to avoid context window bloat
    kb = kb[-15:]
    
    try:
        with open(KB_FILE, "w") as f:
            json.dump(kb, f, indent=2)
    except Exception:
        pass

def read_past_mistakes() -> str:
    """Reads past errors to inject into the generator prompt."""
    if not os.path.exists(KB_FILE):
        return ""
    try:
        with open(KB_FILE, "r") as f:
            kb = json.load(f)
        if not kb:
            return ""
        mistakes = []
        for i, entry in enumerate(kb):
            mistakes.append(f"Historical Error {i+1}: {entry.get('error', '')}")
        return "\n\n".join(mistakes)
    except Exception:
        return ""

def fetch_kaggle_competition_metadata(url: str) -> str:
    """Fetches the title and description from a Kaggle competition webpage.
    
    Args:
        url: The full Kaggle competition URL to fetch metadata for (e.g. https://www.kaggle.com/competitions/titanic).
        
    Returns:
        A string containing the competition title and description.
    """
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown Title"
            
            desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', html, re.IGNORECASE)
            if not desc_match:
                desc_match = re.search(r'<meta\s+property="og:description"\s+content="([^"]+)"', html, re.IGNORECASE)
            desc = desc_match.group(1).strip() if desc_match else "No description available"
            
            return f"Title: {title}\nDescription: {desc}"
    except Exception as e:
        return f"Error fetching webpage: {e}"

def create_agent(name: str, instruction: str, tools: list[Any] = None) -> Agent:
    if tools is None:
        tools = []
    return Agent(
        name=name,
        model=Gemini(model="gemini-flash-latest"),
        input_schema=KaggleState,
        output_schema=KaggleState,
        instruction=instruction,
        tools=tools
    )

competition_ingestion_node = create_agent(
    "competition_ingestion_node",
    (
        "You are a Kaggle expert. Use the fetch_kaggle_competition_metadata tool to read the competition details "
        "using the URL provided in `input_text`. Extract the dataset characteristics (task type, schema, evaluation metric, constraints) "
        "from the fetched metadata. Update ONLY `dataset_characteristics` and retain other fields as they are."
    ),
    tools=[fetch_kaggle_competition_metadata]
)

problem_understanding_node = create_agent(
    "problem_understanding_node",
    "Identify the ML problem type (classification, regression, etc.) and define the objective based on `dataset_characteristics`. Update ONLY `problem_type_patterns` and retain other fields."
)

eda_node = create_agent(
    "eda_node",
    "Perform a lightweight exploratory data analysis plan (missing values, target distribution, feature types, leakage). Append this to `dataset_characteristics` and retain other fields."
)

preprocessing_node = create_agent(
    "preprocessing_node",
    "Generate a basic preprocessing pipeline (imputation, encoding, scaling) suitable for the dataset. Consider any `human_feedback` if provided. Check `past_mistakes` to avoid known API errors. Update `preprocessing_strategy` with descriptions and write the scikit-learn column transformer code in `preprocessing_code`. Retain other fields."
)

from google.adk.tools import google_search

model_research_agent = Agent(
    name="model_research_agent",
    model=Gemini(model="gemini-flash-latest"),
    instruction="Use the google_search tool to research models for the provided machine learning problem context. Return a concise summary of the top recommended models and their scikit-learn equivalents. IMPORTANT: You MUST strictly adhere to any requested model types or constraints specified in the prompt's 'Human Feedback / Constraints' section. If the user asks for Neural Networks, you MUST recommend Neural Networks.",
    tools=[google_search]
)

@node(name="model_research_node", rerun_on_resume=True)
async def model_research_node(ctx: Context, node_input: KaggleState) -> KaggleState:
    prompt = f"Dataset: {node_input.dataset_characteristics}\nProblem: {node_input.problem_type_patterns}"
    if node_input.human_feedback:
        prompt += f"\nHuman Feedback / Constraints: {node_input.human_feedback}"
    res = await ctx.run_node(model_research_agent, node_input=prompt)
    node_input.model_research_notes = extract_text(res)
    return node_input

baseline_model_node = create_agent(
    "baseline_model_node",
    "Select and configure 1-2 strong baseline models based on the research provided in `model_research_notes` for this ML problem. Check `past_mistakes` to avoid known API deprecations. Consider any `human_feedback` provided by the user, especially if they requested specific models. Update `baseline_strategies` with descriptions and write the scikit-learn model instantiation in `baseline_code`. Retain other fields."
)

evaluation_node = create_agent(
    "evaluation_node",
    (
        "Evaluate the trained model conceptually and programmatically. "
        "Select the right split strategy (e.g., StratifiedKFold, TimeSeriesSplit) to prevent target leakage. "
        "Check `past_mistakes` and `human_feedback` to avoid known metric or splitting errors. "
        "Write the scikit-learn cross-validation code to `evaluation_code`. Retain other fields."
    )
)

report_generator_node = create_agent(
    "report_generator_node",
    (
        "You are the final compiler. Produce a final structured summary. Combine the `preprocessing_code`, "
        "`baseline_code`, and `evaluation_code` into a complete runnable script. "
        "IMPORTANT: If `human_feedback` contains verification errors, you MUST analyze the traceback and FIX the bugs "
        "in the final script instead of blindly combining the broken snippets! Check `past_mistakes` to avoid known errors.\n"
        "You MUST include this complete python block (wrapped in markdown code blocks) at the bottom of the `final_report` string, "
        "and save the exact same complete python script in the `final_script` field.\n"
        "Guidelines for Executable Verification:\n"
        "1. The final script MUST be fully self-contained and runnable.\n"
        "2. If `__name__ == '__main__'` is triggered, generate a small synthetic dataset matching the schema, "
        "instantiate pipelines, and execute cross-validation. Do not reference undefined variables like `X_train`.\n"
        "Update ONLY `final_report` and `final_script` and retain all other fields."
    )
)

# Helper function to extract text from raw playground Content input
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

def extract_and_validate_kaggle_url(text: str) -> tuple[str | None, bool]:
    urls = re.findall(r'(https?://[^\s]+)', text)
    for url in urls:
        url = url.rstrip('.,;:)("')
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                continue
            if parsed.username or parsed.password:
                continue
            
            host = parsed.hostname
            if not host:
                continue
            host = host.lower()
            if host != 'kaggle.com' and not host.endswith('.kaggle.com'):
                continue
                
            if not re.search(r'/(competitions|c|datasets)/[\w-]+', parsed.path.lower()):
                continue
                
            ip_str = socket.gethostbyname(host)
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                continue
                
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.getcode() == 200:
                    return url, True
        except Exception:
            continue
            
    path_match = re.search(r'(?:https?://)?(?:www\.)?(kaggle\.com/(?:competitions|c|datasets)/[\w-]+)', text, re.IGNORECASE)
    if path_match:
        fallback_url = "https://www." + path_match.group(1)
        try:
            parsed = urllib.parse.urlparse(fallback_url)
            host = parsed.hostname
            ip_str = socket.gethostbyname(host)
            ip = ipaddress.ip_address(ip_str)
            if not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast):
                req = urllib.request.Request(
                    fallback_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.getcode() == 200:
                        return fallback_url, True
        except Exception:
            pass
            
    return None, False

# Interactive nodes decorated with @node(rerun_on_resume=False)
@node(name="ask_for_url", rerun_on_resume=False)
async def ask_for_url(ctx: Context, message: str = "Please provide a valid Kaggle competition URL to proceed."):
    yield RequestInput(message=message)

@node(name="ask_for_review", rerun_on_resume=False)
async def ask_for_review(ctx: Context, node_input: KaggleState):
    warning = ""
    if not node_input.verification_passed:
        warning = (
            "⚠️ WARNING: The generated code failed automated sandbox verification after 3 attempts!\n"
            f"Final Error Traceback:\n{node_input.verification_error}\n\n"
        )
        
    message = (
        f"{warning}"
        f"Problem Type: {node_input.problem_type_patterns}\n"
        f"Preprocessing: {node_input.preprocessing_strategy}\n"
        f"Models: {node_input.baseline_strategies}\n\n"
        "Do you approve this plan and code? (Reply 'approve' to save to file, or provide feedback for revision)"
    )
    yield RequestInput(message=message)

# Automated Sandbox Execution Verification Node (tests script in-memory/temp files)
@node(name="verify_code_node", rerun_on_resume=True)
async def verify_code_node(ctx: Context, node_input: KaggleState) -> tuple[bool, str]:
    """Executes state.final_script in a temporary validation file to check for errors."""
    if not node_input.final_script.strip():
        return False, "No script found in state.final_script to verify."
        
    temp_path = "temp_validation.py"
    try:
        with open(temp_path, "w") as f:
            f.write(node_input.final_script)
            
        res = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if res.returncode == 0:
            return True, "Execution verified successfully."
        else:
            return False, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, "Execution timed out (10s limit exceeded)."
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"Failed to execute verification sandbox: {str(e)}"

# Deterministic File Writer Node (Only runs after user approval)
@node(name="write_code_file_node", rerun_on_resume=True)
async def write_code_file_node(ctx: Context, node_input: KaggleState) -> str:
    """Writes the approved script from state.final_script to baseline_solution.py."""
    if not node_input.final_script.strip():
        return "No script to write."
    try:
        with open("baseline_solution.py", "w") as f:
            f.write(node_input.final_script)
        return "Successfully wrote baseline_solution.py"
    except Exception as e:
        return f"Failed to write file: {str(e)}"

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
            state = to_state(await ctx.run_node(competition_ingestion_node, state))
            state = to_state(await ctx.run_node(problem_understanding_node, state))
            state = to_state(await ctx.run_node(eda_node, state))
            ctx.state["step"] = "modeling_loop"
            ctx.state["kaggle_state"] = state.model_dump()

        elif ctx.state["step"] == "modeling_loop":
            state.past_mistakes = read_past_mistakes()
            state = to_state(await ctx.run_node(preprocessing_node, state))
            state = to_state(await ctx.run_node(model_research_node, state))
            state = to_state(await ctx.run_node(baseline_model_node, state))
            state = to_state(await ctx.run_node(evaluation_node, state))
            state = to_state(await ctx.run_node(report_generator_node, state))
            
            for attempt in range(3):
                success, err_msg = await ctx.run_node(verify_code_node, state)
                if success:
                    state.verification_passed = True
                    state.verification_error = ""
                    break
                record_sandbox_error(err_msg, state.final_script)
                state.human_feedback = f"CODE VERIFICATION FAILED (Attempt {attempt+1}/3).\n{err_msg}"
                state.past_mistakes = read_past_mistakes()
                state = to_state(await ctx.run_node(report_generator_node, state))
            else:
                state.verification_passed = False
                state.verification_error = err_msg

            ctx.state["step"] = "ask_review"
            ctx.state["kaggle_state"] = state.model_dump()

        elif ctx.state["step"] == "ask_review":
            review_idx = ctx.state.get("review_count", 0)
            interrupt_id = f"review_input_{review_idx}"
            
            if not ctx.state.get("ask_review_yielded", False):
                # FIRST TIME: Yield RequestInput
                ctx.state["ask_review_yielded"] = True
                
                warning = ""
                if not state.verification_passed:
                    warning = f"⚠️ WARNING: Code failed automated sandbox verification!\nFinal Error Traceback:\n{state.verification_error}\n\n"
                
                message = (
                    f"{warning}"
                    f"Problem Type: {state.problem_type_patterns}\n"
                    f"Preprocessing: {state.preprocessing_strategy}\n"
                    f"Models: {state.baseline_strategies}\n\n"
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
            if state.verification_passed:
                await ctx.run_node(write_code_file_node, state)
            else:
                await ctx.run_node(write_code_file_node, state)
            
            from google.genai import types
            yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=state.final_report)]))
            yield Event(output=state.final_report)
            return

# Setup the root agent and expose the App
root_agent = Workflow(
    name="kaggle_copilot_workflow",
    edges=[("START", kaggle_copilot_workflow)]
)

from google.adk.apps import App, ResumabilityConfig

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True)
)
