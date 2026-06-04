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


def classify_failure(norms, grippers, num_steps, movement_threshold, late_ratio):
    if not norms:
        return "no_movement"

    active_steps = sum(x >= movement_threshold for x in norms)
    active_ratio = active_steps / len(norms)
    first_move = first_index(norms, lambda x: x >= movement_threshold)
    first_close = first_index(grippers, lambda x: x is not None and x <= 0.0)

    if active_ratio < 0.1:
        return "no_movement"

    if first_move is not None and first_move > int(num_steps * late_ratio):
        return "timeout_after_late_start"

    if first_close is None:
        return "grasp_failure"

    close_count = sum(x is not None and x <= 0.0 for x in grippers)
    if close_count < max(3, int(num_steps * 0.05)):
        return "grasp_failure"

    if active_ratio > 0.7 and first_close is None:
        return "random_motion"

    return "place_failure"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output", default="results/failure_cases.csv")
    parser.add_argument("--movement_threshold", type=float, default=0.02)
    parser.add_argument("--late_ratio", type=float, default=0.5)
    args = parser.parse_args()

    rows = []

    for episode_dir in find_episode_dirs(args.log_root):
        meta, trace = episode_record(episode_dir)
        if bool(meta.get("success", False)):
            continue

        actions = [action_vector(row) for row in trace]
        norms = [action_norm(action) for action in actions]
        grippers = [gripper_value(action) for action in actions]

        failure_type = classify_failure(
            norms,
            grippers,
            len(trace),
            args.movement_threshold,
            args.late_ratio,
        )

        rows.append(
            {
                "checkpoint": checkpoint_name(meta),
                "task_suite": meta.get("task_suite", ""),
                "task_id": meta.get("task_id", ""),
                "task_description": meta.get("task_description", ""),
                "episode_idx": meta.get("episode_idx", ""),
                "failure_type": failure_type,
                "num_steps": len(trace),
                "mean_action_norm": round(sum(norms) / len(norms), 6) if norms else 0.0,
                "max_action_norm": round(max(norms), 6) if norms else 0.0,
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
            "failure_type",
            "num_steps",
            "mean_action_norm",
            "max_action_norm",
        ],
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
