#!/usr/bin/env python3
"""Run a single EPO-Safe experiment."""

import argparse
import logging
import sys

import anyio

from epo_safe.agents.llm_agent import create_agent
from epo_safe.config import ExperimentConfig
from epo_safe.loop.experiential_loop import ExperientialLoop


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single EPO-Safe experiment")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "side_effects", "off_switch", "absent_supervisor", "boat_race", "whisky_gold",        ],
        help="Environment name",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["static", "cot", "reward_only", "epo_safe", "epo_safe_l0"],
        help="Method/baseline to use",
    )
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per round")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-steps", type=int, default=30, help="Max actions per episode"
    )
    parser.add_argument(
        "--boat-race-actions",
        type=int,
        default=20,
        help="Required number of actions for boat_race (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/", help="Output directory"
    )
    parser.add_argument(
        "--false-positives",
        type=float,
        default=0.0,
        help="False positive rate for safety warnings (default: 0.0)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    config = ExperimentConfig(
        env_name=args.env,
        method=args.method,
        model=args.model,
        episodes_per_round=args.episodes,
        num_rounds=args.rounds,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        boat_race_required_actions=args.boat_race_actions,
        false_positive_rate=args.false_positives,
        output_dir=args.output_dir,
    )

    agent = create_agent(config.model)
    loop = ExperientialLoop(config, agent)
    await loop.run()


if __name__ == "__main__":
    anyio.run(main)
