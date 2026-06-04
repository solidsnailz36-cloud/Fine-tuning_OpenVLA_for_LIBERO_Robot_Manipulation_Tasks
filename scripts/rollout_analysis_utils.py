#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_trace_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_episode_dirs(root):
    root = Path(root)
    return sorted(p.parent for p in root.rglob("trace.jsonl"))


def action_vector(row):
    action = row.get("action", [])
    return [float(x) for x in action]


def action_norm(action, dims=6):
    vals = action[:dims]
    return math.sqrt(sum(x * x for x in vals))


def gripper_value(action):
    if not action:
        return None
    return float(action[-1])


def first_index(values, predicate):
    for idx, value in enumerate(values):
        if predicate(value):
            return idx
    return None


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def episode_record(episode_dir):
    episode_dir = Path(episode_dir)
    meta = read_json(episode_dir / "meta.json")
    trace = read_trace_jsonl(episode_dir / "trace.jsonl")
    return meta, trace


def checkpoint_name(meta):
    value = str(meta.get("checkpoint", "unknown"))
    return Path(value).name or value
