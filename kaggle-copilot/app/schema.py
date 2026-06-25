from pydantic import BaseModel

class KaggleState(BaseModel):
    input_text: str = ""
    dataset_characteristics: str = ""
    problem_type_patterns: str = ""
    preprocessing_strategy: str = ""
    preprocessing_code: str = ""
    feature_engineering_strategy: str = ""
    feature_engineering_code: str = ""
    baseline_strategies: str = ""
    baseline_code: str = ""
    evaluation_code: str = ""
    final_script: str = ""
    human_feedback: str = ""
    final_report: str = ""
    critic_feedback: str = ""
    final_notebook: str = ""
    research_sources: str = ""
