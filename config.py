import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class OptimizationConfig:
    use_llm: bool
    llm_base_url: str
    llm_model: str
    output_dir: str
    agent_models: dict

def load_config() -> OptimizationConfig:
    # Use the directory containing this file as the root for the engine
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    use_llm_str = os.getenv("USE_LLM", "false").lower()
    use_llm = use_llm_str in ("true", "1", "yes")

    # Resolve output dir relative to the root directory
    output_dir_raw = os.getenv("OPTIMIZATION_OUTPUT_DIR", os.path.join("data", "optimization_output"))
    if not os.path.isabs(output_dir_raw):
        output_dir = os.path.join(root_dir, output_dir_raw)
    else:
        output_dir = output_dir_raw

    agent_models = {
        "critic": os.getenv("LLM_MODEL_CRITIC", "deepseek"),
        "developer": os.getenv("LLM_MODEL_DEVELOPER", "qwen"),
        "strategist": os.getenv("LLM_MODEL_STRATEGIST", "mistral"),
    }

    return OptimizationConfig(
        use_llm=use_llm,
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        llm_model=os.getenv("LLM_MODEL", "mistral-nemo-instruct-2407"),
        output_dir=output_dir,
        agent_models=agent_models,
    )

config = load_config()
