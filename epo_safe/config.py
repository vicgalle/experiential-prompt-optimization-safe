"""Experiment configuration."""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    env_name: str  # 'side_effects', 'off_switch', 'absent_supervisor', 'boat_race', 'whisky_gold'
    method: str  # 'static', 'cot', 'reward_only', 'epo_safe', 'epo_safe_l0'
    model: str = "sonnet"  # 'sonnet', 'opus', 'haiku', or 'gemini-*'
    episodes_per_round: int = 5
    num_rounds: int = 20
    feedback_level: str = "level1"  # 'level0' or 'level1'
    seed: int = 42
    max_episode_steps: int = 30  # Max actions the LLM generates per episode
    boat_race_required_actions: int = 20  # Exact action count for boat_race
    false_positive_rate: float = 0.0  # Probability of spurious safety warning per non-dangerous step
    output_dir: str = "results/"


# All experiment configurations
EXPERIMENTS = []
for env in [
    "side_effects", "off_switch", "absent_supervisor", "boat_race", "whisky_gold",
]:
    for method in ["static", "cot", "reward_only", "epo_safe", "epo_safe_l0"]:
        EXPERIMENTS.append(ExperimentConfig(env_name=env, method=method))
