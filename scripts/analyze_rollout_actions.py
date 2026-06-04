#!/usr/bin/env python3

import argparse
from rollout_analysis_utils import (
    action_norm,
    action_vector,
    checkpoint_name,
    episode_record,
    find_episode_dirs,
    first_index,
    gripper_value,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", default="results/latency_summary.csv")
    parser.add_argument("--movement_threshold", type=float, default=0.02)
    parser.add_argument("--gripper_close_threshold", type=float, default=0.0)
    args = parser.parse_args()

    rows = []

    for episode_dir in find_episode_dirs(args.log_root):
        meta, trace = episode_record(episode_dir)
        actions = [action_vector(row) for row in trace]
        norms = [action_norm(action) for action in actions]
        grippers = [gripper_value(action) for action in actions]

        first_move = first_index(norms, lambda x: x >= args.movement_threshold)
        first_gripper_close = first_index(
            grippers,
            lambda x: x is not None and x <= args.gripper_close_threshold,
        )

        mean_norm = sum(norms) / len(norms) if norms else 0.0
        max_norm = max(norms) if norms else 0.0

        rows.append(
            {
                "checkpoint": checkpoint_name(meta),
                "task_suite": meta.get("task_suite", ""),
                "task_id": meta.get("task_id", ""),
                "task_description": meta.get("task_description", ""),
                "episode_idx": meta.get("episode_idx", ""),
                "success": bool(meta.get("success", False)),
                "num_steps": len(trace),
                "first_action_timestep": first_move if first_move is not None else "",
                "first_gripper_close_timestep": first_gripper_close if first_gripper_close is not None else "",
                "mean_action_norm": round(mean_norm, 6),
                "max_action_norm": round(max_norm, 6),
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
            "episode_idx",
            "success",
            "num_steps",
            "first_action_timestep",
            "first_gripper_close_timestep",
            "mean_action_norm",
            "max_action_norm",
        ],
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
