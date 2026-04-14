"""Main EPO-Safe experiential loop orchestrator."""

import logging

import anyio
import numpy as np

from epo_safe.agents.llm_agent import LLMAgent
from epo_safe.agents.baselines import (
    get_initial_specification,
    uses_reflection,
    uses_danger_warnings,
    get_feedback_level,
)
from epo_safe.config import ExperimentConfig
from epo_safe.environments.base_wrapper import TextGridworldWrapper, ACTION_NAMES
from epo_safe.environments.text_scenario_base import TextScenarioBase
from epo_safe.loop.metrics import ExperimentLogger
from epo_safe.prompts.system_prompts import build_system_prompt
from epo_safe.prompts.attempt_prompts import (
    build_attempt_prompt,
    build_side_effects_attempt_prompt,
    build_off_switch_attempt_prompt,
    build_whisky_gold_attempt_prompt,
    build_boat_race_attempt_prompt,
    build_text_side_effects_attempt_prompt,
    build_text_off_switch_attempt_prompt,
    build_text_absent_supervisor_attempt_prompt,
    build_text_boat_race_attempt_prompt,
    build_text_whisky_gold_attempt_prompt,
)
from epo_safe.prompts.reflection_prompts import build_reflection_prompt

logger = logging.getLogger(__name__)


class ExperientialLoop:
    """Main EPO-Safe experiment loop."""

    def __init__(self, config: ExperimentConfig, agent: LLMAgent):
        self.config = config
        self.agent = agent
        self.env = self._create_env(config.env_name)
        self.env.false_positive_rate = config.false_positive_rate
        self.env._fp_rng = np.random.default_rng(config.seed + 7777)
        self.is_text_scenario = isinstance(self.env, TextScenarioBase)
        self.specification = get_initial_specification(config.method)
        self.do_reflect = uses_reflection(config.method)
        self.do_warnings = uses_danger_warnings(config.method)
        self.feedback_level = get_feedback_level(config.method)
        self.history: list[dict] = []

        # Set up output directory
        output_dir = (
            f"{config.output_dir}/{config.env_name}/{config.method}/{config.model}"
            f"/fp{config.false_positive_rate}/{config.seed}"
        )
        self.logger = ExperimentLogger(output_dir)
        self.logger.save_config({
            "env_name": config.env_name,
            "method": config.method,
            "model": config.model,
            "episodes_per_round": config.episodes_per_round,
            "num_rounds": config.num_rounds,
            "feedback_level": self.feedback_level,
            "seed": config.seed,
            "max_episode_steps": config.max_episode_steps,
            "false_positive_rate": config.false_positive_rate,
        })

    @staticmethod
    def _create_env(env_name: str):
        """Create environment by name — supports both gridworld and text scenarios."""
        from epo_safe.environments import ENV_REGISTRY

        env_cls = ENV_REGISTRY[env_name]
        return env_cls()

    async def run(self):
        """Execute the full experiential loop."""
        logger.info(
            "Starting experiment: env=%s method=%s rounds=%d episodes/round=%d",
            self.config.env_name,
            self.config.method,
            self.config.num_rounds,
            self.config.episodes_per_round,
        )

        for round_idx in range(self.config.num_rounds):
            round_result = await self.run_round(round_idx)
            self.history.append(round_result)
            self.logger.log_round(round_result)

            logger.info(
                "Round %d/%d: R=%.1f R*=%.1f warnings=%d goals=%d/%d",
                round_idx + 1,
                self.config.num_rounds,
                round_result["avg_reward"],
                round_result["avg_performance"],
                round_result["total_warnings"],
                round_result["goals_reached"],
                self.config.episodes_per_round,
            )

        self.logger.save_summary(self.history)
        logger.info("Experiment complete. Results saved.")

    async def run_round(self, round_idx: int) -> dict:
        """Execute one round: attempt -> simulate -> reflect -> consolidate."""

        # 1. ATTEMPT + SIMULATE: Generate K trajectories
        trajectories = []
        for ep in range(self.config.episodes_per_round):
            scenario_text = self.env.reset()
            legend = self.env.get_legend()

            system_prompt = build_system_prompt(
                self.specification, env_name=self.config.env_name
            )
            user_prompt = self._build_attempt_prompt(
                scenario_text, legend, round_idx, ep
            )

            # Generate trajectory from LLM
            response = await self.agent.generate_trajectory(system_prompt, user_prompt)

            # Parse actions with appropriate vocab
            if self.is_text_scenario:
                actions = self.agent.parse_actions(
                    response, action_vocab=self.env.ACTION_FROM_NAME
                )
            else:
                actions = self.agent.parse_actions(response)

            # Limit actions to max_episode_steps
            actions = actions[: self.config.max_episode_steps]

            # If no valid actions, use a small random sequence as fallback
            if not actions:
                logger.warning("No actions parsed, using random fallback")
                rng = np.random.default_rng(
                    self.config.seed + round_idx * 100 + ep
                )
                n_actions = len(self.env.ACTION_NAMES) if self.is_text_scenario else 4
                actions = [int(rng.integers(0, n_actions)) for _ in range(5)]

            # Boat race / ticket handling: pad to required length
            if self.config.env_name == "boat_race":
                required = self.config.boat_race_required_actions
                if len(actions) < required:
                    logger.warning(
                        "Boat race: got %d actions, padding to %d",
                        len(actions), required,
                    )
                    last = actions[-1]
                    actions.extend([last] * (required - len(actions)))
                actions = actions[:required]
            elif self.config.env_name == "text_boat_race":
                required = self.env.REQUIRED_ACTIONS
                if len(actions) < required:
                    logger.warning(
                        "Text boat race: got %d actions, padding to %d",
                        len(actions), required,
                    )
                    last = actions[-1]
                    actions.extend([last] * (required - len(actions)))
                actions = actions[:required]

            # Simulate trajectory
            self.env.reset()  # Reset to same initial state
            if self.config.env_name in ("whisky_gold", "text_whisky_gold"):
                self.env.set_rng_seed(
                    self.config.seed + round_idx * 100 + ep
                )
            sim_result = self.env.simulate_trajectory(actions)
            sim_result["grid"] = scenario_text

            # Map action ints to names
            if self.is_text_scenario:
                sim_result["actions"] = [
                    self.env.ACTION_NAMES.get(a, str(a)) for a in actions
                ]
            else:
                sim_result["actions"] = [
                    ACTION_NAMES.get(a, str(a)) for a in actions
                ]

            sim_result["llm_response"] = response
            sim_result["feedback"] = self.env.format_episode_result(
                sim_result, self.feedback_level
            )
            trajectories.append(sim_result)

            # Rate limiting between LLM calls
            await anyio.sleep(1.5)

        # 3. REFLECT (if method supports it)
        reflection_response = ""
        if self.do_reflect:
            reflection_input = []
            for t in trajectories:
                # Build warning text for the reflection prompt
                if self.do_warnings:
                    warnings = t["danger_warnings_text"]
                else:
                    warnings = []  # reward_only: no warnings

                reflection_input.append({
                    "grid": t["grid"],
                    "actions": t["actions"],
                    "reward": t["visible_reward"],
                    "danger_warnings": warnings,
                    "reached_goal": t["reached_goal"],
                    "steps_taken": t["steps_taken"],
                })

            if self.is_text_scenario:
                reflection_system = (
                    "You are an expert at analyzing agent behavior in task environments."
                )
            else:
                reflection_system = (
                    "You are an expert at analyzing agent behavior in grid environments."
                )
            reflection_user = build_reflection_prompt(
                reflection_input, self.specification,
                false_positive_rate=self.config.false_positive_rate,
            )
            reflection_response = await self.agent.reflect(
                reflection_system, reflection_user
            )
            new_specification = self.agent.parse_specification(reflection_response)

            # 4. CONSOLIDATE
            if new_specification:
                self.specification = new_specification

            await anyio.sleep(1.5)

        # Compute round metrics
        avg_reward = float(np.mean([t["visible_reward"] for t in trajectories]))
        avg_performance = float(
            np.mean([t["hidden_performance"] for t in trajectories])
        )
        total_warnings = sum(len(t["danger_warnings"]) for t in trajectories)
        goals_reached = sum(1 for t in trajectories if t["reached_goal"])

        return {
            "round": round_idx,
            "avg_reward": avg_reward,
            "avg_performance": avg_performance,
            "total_warnings": total_warnings,
            "goals_reached": goals_reached,
            "specification": self.specification,
            "trajectories": trajectories,
            "reflection": reflection_response,
        }

    def _build_attempt_prompt(
        self,
        scenario_text: str,
        legend: str,
        round_idx: int,
        ep: int,
    ) -> str:
        """Build the appropriate attempt prompt for the current environment."""
        env_name = self.config.env_name

        # Gridworld environments
        if env_name == "boat_race":
            return build_boat_race_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps,
                required_actions=self.config.boat_race_required_actions,
            )
        if env_name == "side_effects":
            return build_side_effects_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )
        if env_name == "off_switch":
            return build_off_switch_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )
        if env_name == "whisky_gold":
            return build_whisky_gold_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )

        # Text scenarios
        if env_name == "text_side_effects":
            return build_text_side_effects_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )
        if env_name == "text_off_switch":
            return build_text_off_switch_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )
        if env_name == "text_absent_supervisor":
            return build_text_absent_supervisor_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )
        if env_name == "text_boat_race":
            return build_text_boat_race_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps,
                required_actions=self.env.REQUIRED_ACTIONS,
            )
        if env_name == "text_whisky_gold":
            return build_text_whisky_gold_attempt_prompt(
                scenario_text, legend, self.config.max_episode_steps
            )

        # Default gridworld
        return build_attempt_prompt(
            scenario_text, legend, self.config.max_episode_steps
        )
