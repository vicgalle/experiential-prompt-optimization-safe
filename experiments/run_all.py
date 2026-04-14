#!/usr/bin/env python3
"""Run all EPO-Safe experiments sequentially."""

import argparse
import logging

import anyio

from epo_safe.agents.llm_agent import create_agent
from epo_safe.config import EXPERIMENTS, ExperimentConfig
from epo_safe.loop.experiential_loop import ExperientialLoop


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all EPO-Safe experiments")
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per round")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds")
    parser.add_argument(
        "--output-dir", type=str, default="results/", help="Output directory"
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="*",
        default=None,
        help="Only run these environments (default: all)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Only run these methods (default: all)",
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

    experiments = EXPERIMENTS
    if args.envs:
        experiments = [e for e in experiments if e.env_name in args.envs]
    if args.methods:
        experiments = [e for e in experiments if e.method in args.methods]

    total = len(experiments)
    logger.info("Running %d experiments", total)

    for i, base_config in enumerate(experiments):
        config = ExperimentConfig(
            env_name=base_config.env_name,
            method=base_config.method,
            model=args.model,
            episodes_per_round=args.episodes,
            num_rounds=args.rounds,
            seed=base_config.seed,
            false_positive_rate=args.false_positives,
            output_dir=args.output_dir,
        )

        logger.info(
            "=== Experiment %d/%d: %s / %s ===",
            i + 1,
            total,
            config.env_name,
            config.method,
        )

        try:
            agent = create_agent(config.model)
            loop = ExperientialLoop(config, agent)
            await loop.run()
        except Exception:
            logger.exception(
                "Experiment failed: %s / %s", config.env_name, config.method
            )
            continue

    logger.info("All experiments complete.")


if __name__ == "__main__":
    anyio.run(main)
