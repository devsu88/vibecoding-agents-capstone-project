"""
Agents Module

This module defines all the LLM-powered agents used in the Kaggle Copilot.
Each agent is configured with a specific persona, instructions, and optionally, tools.
They all share the same input/output schema (`KaggleState`) to ensure seamless
data passing throughout the orchestration workflow.
"""

from typing import Any
from google.adk.agents import Agent
from google.adk.models import Gemini
from app.schema import KaggleState
from app.tools import fetch_kaggle_competition_metadata, custom_web_search

def create_agent(name: str, instruction: str, tools: list[Any] = None) -> Agent:
    """
    Helper function to instantiate an ADK Agent with standard configurations.

    Args:
        name (str): The unique identifier name for the agent.
        instruction (str): The system prompt/instruction detailing the agent's behavior.
        tools (list[Any], optional): A list of callable python functions the agent can use.

    Returns:
        Agent: A fully configured ADK Agent instance.
    """
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

# ---------------------------------------------------------------------------
# Data Understanding Agents
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Feature Engineering & Preprocessing Agents
# ---------------------------------------------------------------------------

preprocessing_node = create_agent(
    "preprocessing_node",
    "Generate a basic preprocessing pipeline (imputation, encoding, scaling) suitable for the dataset. Consider any `human_feedback` if provided. Update `preprocessing_strategy` with descriptions and write the scikit-learn column transformer code in `preprocessing_code`. Retain other fields."
)

feature_engineering_node = create_agent(
    "feature_engineering_node",
    "Based on `dataset_characteristics` and `problem_type_patterns`, propose domain-specific feature engineering (e.g., aggregations, date parts, interactions, target encoding). Write the python code extending the preprocessing pipeline or as standalone pandas logic in `feature_engineering_code` and describe the strategy in `feature_engineering_strategy`. Consider `human_feedback`. Retain other fields."
)

# ---------------------------------------------------------------------------
# Modeling & Research Agents
# ---------------------------------------------------------------------------

model_research_agent = Agent(
    name="model_research_agent",
    model=Gemini(model="gemini-flash-latest"),
    instruction=(
        "Use the custom_web_search tool to research the best modeling approaches for the provided ML problem context. "
        "Then, decide on a modeling strategy (e.g., 1-2 strong baseline models, or a combination like Voting/Stacking ensemble). "
        "For tabular data, strongly prefer Gradient Boosting libraries (LightGBM, XGBoost, CatBoost). Consider `human_feedback`. "
        "IMPORTANT: You MUST include a 'SOURCES:' section at the very end of your response listing the sources you consulted. "
        "Format each source as a markdown link: `- [Title](URL)`. Return a concise strategy summary."
    ),
    tools=[custom_web_search]
)

baseline_model_agent = create_agent(
    "baseline_model_agent",
    "Based on the provided research results (appended to `problem_type_patterns`), write the complete model instantiation and ensemble code in `baseline_code`. Update `baseline_strategies` with the chosen strategy. Retain other fields."
)

evaluation_node = create_agent(
    "evaluation_node",
    (
        "Evaluate the trained model conceptually and programmatically. "
        "Select the right split strategy (e.g., StratifiedKFold, TimeSeriesSplit) to prevent target leakage. "
        "Check `human_feedback` to avoid known metric or splitting errors. "
        "Write the scikit-learn cross-validation code to `evaluation_code`. Retain other fields."
    )
)

# ---------------------------------------------------------------------------
# Compilation & Review Agents
# ---------------------------------------------------------------------------

report_generator_node = create_agent(
    "report_generator_node",
    (
        "You are the final compiler. Produce a final structured summary. Combine the `preprocessing_code`, "
        "`feature_engineering_code`, `baseline_code`, and `evaluation_code` into a complete runnable script. "
        "IMPORTANT: Integrate any changes requested in `human_feedback` instead of blindly combining the snippets.\n"
        "You MUST include this complete python block (wrapped in markdown code blocks) at the bottom of the `final_report` string, "
        "and save the exact same complete python script in the `final_script` field.\n"
        "Guidelines for the Final Script:\n"
        "1. The final script MUST be fully self-contained and runnable.\n"
        "2. If `__name__ == '__main__'` is triggered, write code to load the actual dataset from 'train.csv' and 'test.csv' using pandas (e.g., `pd.read_csv('train.csv')`). Do not generate synthetic data.\n"
        "Update ONLY `final_report` and `final_script` and retain all other fields."
    )
)

code_critic_node = create_agent(
    "code_critic_node",
    "You are a Senior Kaggle Grandmaster acting as a Code Critic. Review the generated `final_script` and `final_report` for common pitfalls like data leakage, weak cross-validation, missing imports, or inefficient Pandas operations. If you find issues, write a brief, constructive 'Critic's Note' in `critic_feedback` (max 2-3 sentences). If the code is perfect, leave `critic_feedback` empty. Do NOT modify the code yourself. Retain other fields."
)
