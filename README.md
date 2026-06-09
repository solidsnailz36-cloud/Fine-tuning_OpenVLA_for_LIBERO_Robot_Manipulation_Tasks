#  Fine-tuning OpenVLA for LIBERO Robot Manipulation Tasks

This repository enables fine-tuning of **OpenVLA** (Vision-Language-Action) models on **LIBERO** robot manipulation tasks, with comprehensive tools for model training and detailed rollout evaluation.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n openvla-libero python=3.10 -y
conda activate openvla-libero

# Install dependencies
pip install -r requirements.txt

# Install local packages
pip install -e ./LIBERO
pip install -e ./openvla
```

**Dependencies**: PyTorch, Transformers, PEFT, robosuite, MuJoCo, Weights & Biases, and others (see [requirements.txt](requirements.txt) and [environment.yml](environment.yml))

### 2. Download Data

Place the LIBERO dataset in the `datasets/` directory:

```bash
datasets/
└── libero_spatial_no_noops/  # Download LIBERO dataset here
```

Download LIBERO from the official repository if not already present.

### 3. Fine-tune Model

Use the training script to fine-tune OpenVLA on LIBERO tasks:

```bash
bash openvla/train_libero.sh
```

This generates:
- Checkpoints in `datadisk/checkpoints/`
- LoRA adapters in `datadisk/adapter_tmp/`

See [openvla/train_libero.sh](openvla/train_libero.sh) for configuration details.

### 4. Evaluate Model

Run evaluation with the checkpoint:

```bash
python openvla/run_libero_eval.py \
    --checkpoint <path_to_checkpoint> \
    --output_dir results/
```

See [openvla/run_libero_eval.py](openvla/run_libero_eval.py) for more options.

## Rollout Analysis & Evaluation Suite ⭐

The most valuable part of this project is the **comprehensive rollout evaluation toolkit** in [scripts/](scripts/). These tools generate detailed assessments and visualizations of model behavior:

### Core Analysis Scripts

| Script | Purpose |
|--------|---------|
| [scripts/summarize_results.py](scripts/summarize_results.py) | Aggregate success rates across rollouts; generates summary CSV with checkpoint, task suite, task ID, and success metrics |
| [scripts/analyze_rollout_actions.py](scripts/analyze_rollout_actions.py) | Extract and analyze action sequences from rollouts; plot action trajectories over time |
| [scripts/plot_action_traces.py](scripts/plot_action_traces.py) | Visualize action norms and gripper states; generate diagnostic plots for each episode |
| [scripts/classify_failures.py](scripts/classify_failures.py) | Categorize failure modes; generate failure case analysis with statistics |

### Utility Functions

[scripts/rollout_analysis_utils.py](scripts/rollout_analysis_utils.py) provides helper functions:
- `episode_record()`: Parse episode metadata and results
- `action_vector()`, `action_norm()`: Action sequence processing
- `find_episode_dirs()`: Traverse rollout directory structure
- `write_csv()`: Format results for analysis

### Workflow Example

```bash
# 1. Summarize overall success rates
python scripts/summarize_results.py --log_root experiments/logs/ --output results/summary.csv

# 2. Analyze action traces and generate plots
python scripts/analyze_rollout_actions.py --log_root experiments/logs/

# 3. Classify and categorize failures
python scripts/classify_failures.py --log_root experiments/logs/

# 4. Results and visualizations
ls results/
```

Results are organized in `results/` with:
- `failure_cases.csv` - Detailed failure categorization
- `latency_summary.csv` - Timing and performance metrics  
- Plots and visualizations for action traces

## Directory Structure

```
Fine-tuning_OpenVLA_for_LIBERO_Robot_Manipulation_Tasks/
├── openvla/                         # OpenVLA training code
│   ├── train_libero.sh             # Training launcher
│   └── run_libero_eval.py          # Evaluation script
├── scripts/                         # Rollout analysis & evaluation tools ⭐
│   ├── summarize_results.py
│   ├── analyze_rollout_actions.py
│   ├── plot_action_traces.py
│   ├── classify_failures.py
│   └── rollout_analysis_utils.py
├── LIBERO/                          # LIBERO benchmark environment
├── datadisk/
│   ├── checkpoints/                 # Model checkpoints
│   └── adapter_tmp/                 # LoRA adapters
├── datasets/                        # LIBERO datasets
├── experiments/                     # Experiment logs & results
├── results/                         # Analysis results & summaries
├── rollouts/                        # Rollout recordings by date
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment config
└── README.md
```

## Model Weights & Checkpoints

- Model checkpoints are stored in `datadisk/checkpoints/` (not included in repo)
- LoRA adapters in `datadisk/adapter_tmp/` can be preserved:
  - `adapter_model.safetensors`
  - `adapter_config.json`
- Base models from Hugging Face can be re-downloaded as needed
- For large file storage, consider Hugging Face Hub, Google Drive, or Git LFS

## Notes

- **GPU recommended** for training and evaluation
- **CPU suitable** for code inspection, lightweight debugging, and result analysis
- Acceleration packages (flash-attn, bitsandbytes) require compatible CUDA setup
- Large raw datasets not included in repository

## Citation

```bibtex
@misc{openvla,
  title={OpenVLA},
  author={OpenVLA Authors},
  year={2024},
  howpublished={\url{https://github.com/openvla/openvla}}
}

@misc{libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={LIBERO Authors},
  year={2023},
  howpublished={\url{https://github.com/Lifelong-Robot-Learning/LIBERO}}
}
```
