#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


def copy_if_exists(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            print(f"[copy] {src} -> {dst_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_dir",
        default="/root/gpufree-data/project/datadisk/adapter_tmp/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r16+dropout-0.0--image_aug",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/gpufree-data/project/datadisk/checkpoints/remerged-openvla-libero-spatial-image-aug",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Optional override. If omitted, read base_model_name_or_path from adapter_config.json.",
    )
    parser.add_argument("--max_shard_size", default="5GB")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {adapter_config_path}")

    with adapter_config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("No base model found. Pass --base_model explicitly.")

    base_model_path = Path(base_model).expanduser()
    print(f"[info] adapter_dir: {adapter_dir}")
    print(f"[info] base_model:  {base_model}")
    print(f"[info] output_dir:  {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output dir already exists and is not empty: {output_dir}\n"
            "Use a fresh directory to avoid mixing old checkpoint files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] loading base config")
    config = AutoConfig.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    print("[load] loading base OpenVLA model")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    print("[load] loading LoRA adapter")
    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir),
        torch_dtype=torch.bfloat16,
        is_trainable=False,
    )

    print("[merge] merge_and_unload")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    print("[save] saving merged model")
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    print("[save] saving processor/tokenizer")
    try:
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        processor.save_pretrained(str(output_dir))
    except Exception as e:
        print(f"[warn] AutoProcessor save failed: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer.save_pretrained(str(output_dir))
    except Exception as e:
        print(f"[warn] AutoTokenizer save failed: {e}")

    # LIBERO eval needs dataset/action normalization stats. Prefer adapter dir if present;
    # otherwise copy from the old full checkpoint if you pass it later manually.
    copy_if_exists(
        adapter_dir,
        output_dir,
        [
            "dataset_statistics.json",
            "preprocessor_config.json",
            "generation_config.json",
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )

    print("[done] remerged checkpoint written to:")
    print(output_dir)


if __name__ == "__main__":
    main()

