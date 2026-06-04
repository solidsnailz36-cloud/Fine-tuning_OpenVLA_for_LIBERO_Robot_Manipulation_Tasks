#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 可选：不登录 wandb
# export WANDB_MODE=offline

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29505
export TOKENIZERS_PARALLELISM=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/root/gpufree-data/project/datasets/openvla-libero-spatial" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "/root/gpufree-data/project/datadisk/checkpoints" \
  --adapter_tmp_dir "/root/gpufree-data/project/datadisk/adapter_tmp" \
  --batch_size 1 \
  --max_steps 5000 \
  --save_steps 1000 \
  --learning_rate 5e-4 \
  --grad_accumulation_steps 16 \
  --image_aug True \
  --use_lora True \
  --lora_rank 16 \
  --lora_dropout 0.0 \
  --use_quantization False \
  --wandb_project "openvla-finetune_take_3" \
  --wandb_entity "solidsnailz36-shandong-university"

