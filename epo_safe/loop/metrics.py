"""Logging, metrics, and analysis utilities."""

import json
import os
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Logs experiment data per round to disk."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_path = self.output_dir / "rounds.jsonl"
        self.specs_path = self.output_dir / "specifications.jsonl"
        self.config_path = self.output_dir / "config.json"

    def save_config(self, config: dict):
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def log_round(self, round_result: dict):
        """Append one round of results to the JSONL file."""
        # Create a serializable copy (strip non-serializable data)
        record = {
            "round": round_result["round"],
            "avg_reward": round_result["avg_reward"],
            "avg_performance": round_result["avg_performance"],
            "total_warnings": round_result["total_warnings"],
            "goals_reached": round_result["goals_reached"],
            "specification": round_result["specification"],
            "reflection": round_result.get("reflection", ""),
        }

        # Log trajectory summaries (not full state lists)
        traj_summaries = []
        for t in round_result.get("trajectories", []):
            traj_summaries.append({
                "actions": t.get("actions", []),
                "visible_reward": t.get("visible_reward", 0),
                "hidden_performance": t.get("hidden_performance", 0),
                "reached_goal": t.get("reached_goal", False),
                "interrupted": t.get("interrupted", False),
                "steps_taken": t.get("steps_taken", 0),
                "danger_warnings_text": t.get("danger_warnings_text", []),
            })
        record["trajectories"] = traj_summaries

        with open(self.rounds_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Log specification separately
        spec_record = {
            "round": round_result["round"],
            "specification": round_result["specification"],
        }
        with open(self.specs_path, "a") as f:
            f.write(json.dumps(spec_record) + "\n")

    def save_summary(self, history: list[dict]):
        """Save final summary statistics."""
        if not history:
            return

        summary = {
            "num_rounds": len(history),
            "final_avg_reward": history[-1]["avg_reward"],
            "final_avg_performance": history[-1]["avg_performance"],
            "final_specification": history[-1]["specification"],
            "reward_trajectory": [h["avg_reward"] for h in history],
            "performance_trajectory": [h["avg_performance"] for h in history],
            "warning_trajectory": [h["total_warnings"] for h in history],
            "goals_trajectory": [h["goals_reached"] for h in history],
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def load_experiment_results(results_dir: str) -> list[dict]:
    """Load all rounds from a rounds.jsonl file."""
    rounds_path = Path(results_dir) / "rounds.jsonl"
    if not rounds_path.exists():
        return []
    rounds = []
    with open(rounds_path) as f:
        for line in f:
            if line.strip():
                rounds.append(json.loads(line))
    return rounds


def load_summary(results_dir: str) -> dict | None:
    """Load experiment summary."""
    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)
