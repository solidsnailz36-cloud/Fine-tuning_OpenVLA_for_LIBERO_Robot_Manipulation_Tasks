#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rollout_analysis_utils import (
    action_norm,
    action_vector,
    checkpoint_name,
    episode_record,
    find_episode_dirs,
    gripper_value,
)


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def plot_action_norm_and_gripper(xs, norms, grippers, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 3.5))

    color_norm = "tab:blue"
    color_gripper = "tab:red"

    # 左 y 轴：action norm
    line1 = ax1.plot(
        xs,
        norms,
        color=color_norm,
        linewidth=1.6,
        label="arm action norm",
    )
    ax1.set_xlabel("timestep")
    ax1.set_ylabel("arm action norm", color=color_norm)
    ax1.tick_params(axis="y", labelcolor=color_norm)
    ax1.grid(True, alpha=0.3)

    # 右 y 轴：gripper
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        xs,
        grippers,
        color=color_gripper,
        linewidth=1.3,
        alpha=0.85,
        label="gripper",
    )
    ax2.set_ylabel("gripper action", color=color_gripper)
    ax2.tick_params(axis="y", labelcolor=color_gripper)

    # 如果 gripper 是 -1/1 或 0/1，固定一下 y 轴范围，方便不同图之间比较
    valid_grippers = []
    for g in grippers:
        try:
            if np.isfinite(g):
                valid_grippers.append(float(g))
        except Exception:
            pass

    if valid_grippers:
        gmin = min(valid_grippers)
        gmax = max(valid_grippers)

        if gmin >= -1.1 and gmax <= 1.1:
            ax2.set_ylim(-1.2, 1.2)

    # 合并两个 y 轴的 legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    combined_dir = output_dir / "combined_action_gripper_plots"
    combined_dir.mkdir(parents=True, exist_ok=True)


    for episode_dir in find_episode_dirs(args.log_root):
        meta, trace = episode_record(episode_dir)
        actions = [action_vector(row) for row in trace]
        norms = [action_norm(action) for action in actions]
        grippers = [gripper_value(action) for action in actions]
        xs = list(range(len(trace)))

        ckpt = safe_name(checkpoint_name(meta))
        task_id = safe_name(meta.get("task_id", "task"))
        episode_idx = safe_name(meta.get("episode_idx", "episode"))
        success = "success" if bool(meta.get("success", False)) else "failure"

        filename = f"{ckpt}_task_{task_id}_episode_{episode_idx}_{success}.png"

        title_prefix = f"{ckpt} | task {task_id} | episode {episode_idx} | {success}"

        plot_action_norm_and_gripper(
            xs,
            norms,
            grippers,
            title=f"{title_prefix} | action norm + gripper",
            output_path=combined_dir / filename,
        )

    print(f"Wrote plots to {combined_dir}")


if __name__ == "__main__":
    main()
