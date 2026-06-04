#!/usr/bin/env python3

import argparse
from collections import defaultdict
from rollout_analysis_utils import (
    checkpoint_name,
    episode_record,
    find_episode_dirs,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", default="results/results_summary.csv")
    args = parser.parse_args()

    stats = defaultdict(lambda: {"episodes": 0, "successes": 0})

    for episode_dir in find_episode_dirs(args.log_root):
        meta, _ = episode_record(episode_dir)
        key = (
            checkpoint_name(meta),
            meta.get("task_suite", ""),
            meta.get("task_id", ""),
            meta.get("task_description", ""),
        )
        stats[key]["episodes"] += 1
        stats[key]["successes"] += int(bool(meta.get("success", False)))

    rows = []
    for key, value in sorted(stats.items()):
        checkpoint, task_suite, task_id, task_description = key
        episodes = value["episodes"]
        successes = value["successes"]
        rows.append(
            {
                "checkpoint": checkpoint,
                "task_suite": task_suite,
                "task_id": task_id,
                "task_description": task_description,
                "episodes": episodes,
                "successes": successes,
                "success_rate": round(successes / episodes, 4) if episodes else 0.0,
            }
        )

    write_csv(
        args.output,
        rows,
        [
            "checkpoint",
            "task_suite",
            "task_id",
            "task_description",
            "episodes",
            "successes",
            "success_rate",
        ],
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
