# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks

Official repository for the paper SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks.

## Announcement

**2026.4.7: This paper has been accepted to ACL 2026 Main Conference.**

## Quick Start

### 1. Environment Setup

First, run the environment installation script. This will:
- Install UV package manager
- Create a Python 3.10 virtual environment
- Install vLLM, SGLang, and other dependencies
- Install all required packages for the project

```bash
bash uv_verl.sh
```

### 2. Run Training

After the environment setup is complete, choose and run the corresponding training script:

#### DeepSeek-R1-Distill-Qwen 1.5B SPPO DeepscaleR Training
```bash
bash run_scripts/run_ds1.5B_PPO_SEQUENCE_shuffle.sh
```

#### DeepSeek-R1-Distill-Qwen 7B DAPO-17k Training
```bash
bash run_scripts/run_R1-7B_DAPO_SEQUENCE.sh
```

#### DeepSeek-R1-Distill-Qwen 7B DAPO-17k  with Small Critic Training
```bash
bash run_scripts/run_R1-7B_DAPO_SEQUENCE_small_critic.sh
```

## Project Structure

```
.
├── data/                    # Training and evaluation data
├── verl/                    # Core library
├── run_scripts/             # Training launch scripts
├── scripts/                 # Utility scripts
└── uv_verl.sh              # Environment setup script
```

## Data Preparation

Ensure the following data files are available:
- `data/deepscaler-math.parquet` - Training data for 1.5B model
- `data/dapo-math-17k_dedup.parquet` - Training data for 7B model
- `data/offline_eval/math__aime_repeated_8x_240.parquet` - AIME24 test set
- `data/offline_eval/math__math_500.parquet` - MATH test set
- `data/offline_eval/math__amc23_2025.parquet` - AMC23 test set
- `data/offline_eval/math__aime2025_2025.parquet` - AIME25 test set
- `data/offline_eval/math__minerva_math_2025_processed.parquet` - MINERVA test set


## Notes

- First-time run will download models, ensure you have a stable internet connection
- Multi-GPU environment is recommended for better training performance
- Training logs and checkpoints will be saved in the working directory
