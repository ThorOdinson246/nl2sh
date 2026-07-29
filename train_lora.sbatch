#!/bin/bash
# ---------------------------------------------------------------------------
# NL2SH LoRA fine-tune on gpu003 (A100).
#
# SUBMIT LIKE THIS -- the mkdir MUST happen before sbatch, because SLURM opens
# the --output/--error paths at submit time and the job dies instantly if the
# directory does not exist:
#
#   mkdir -p /homes/01/mxpoudel/Grad/nl2sh/slurm_outputs
#   sbatch /homes/01/mxpoudel/Grad/nl2sh/train_lora.sbatch
#
# Log paths are absolute and on the shared NFS home so they are writable from
# gpu003. Never use $TMPDIR or a relative path for SLURM logs here.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nl2sh_lora
#SBATCH --partition=suliaoma
#SBATCH --qos=bxmarg
#SBATCH --nodelist=gpu003
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
# Generous budget: packing=False (see below) means no sequence-packing
# amortization, so real wall-clock is materially higher than a packed run.
# bxmarg QOS has no wall-clock limit; erring generous costs nothing but
# calendar time, and accuracy matters more than runtime for this project.
#SBATCH --output=/homes/01/mxpoudel/Grad/nl2sh/slurm_outputs/train_%x_%j.out
#SBATCH --error=/homes/01/mxpoudel/Grad/nl2sh/slurm_outputs/train_%x_%j.err

set -euo pipefail

PROJ=/homes/01/mxpoudel/Grad/nl2sh
PIXI=/homes/01/mxpoudel/Programs/gdal/.pixi/envs/default
VENV=$PROJ/venv
PY=$VENV/bin/python                 # venv built FROM $PIXI/bin/python, see header note

# Do NOT source conda/pixi activate scripts; call the interpreter directly.
# Keep the pixi env's shared libs (libstdc++, libgomp) visible to the venv:
export LD_LIBRARY_PATH=$PIXI/lib:${LD_LIBRARY_PATH:-}

# Offline-safe caches on shared FS (pre-populate from the login node).
export HF_HOME=/homes/01/mxpoudel/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
# If the preflight showed no outbound HTTPS from gpu003, uncomment:
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
RUN=${SLURM_JOB_NAME}_$(basename "$MODEL")_${SLURM_JOB_ID}
OUT=$PROJ/runs/$RUN
mkdir -p "$OUT"

echo "[$(date)] host=$(hostname) job=$SLURM_JOB_ID model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# packing=False, deliberately: packing=True disables completion-only
# masking entirely in this TRL version (train_lora.py prints this as an
# explicit warning, not a silent gotcha). That means training loss would
# include the instruction/system-prompt tokens, working directly against
# the "+21% from correct output-format learning" result this whole approach
# is built on (../hist/docs/GENERATION_FEASIBILITY.md,
# docs/DISTILLATION.md sec 8). For this first real baseline, correct
# masking matters more than the wall-clock savings from packing.
srun $PY $PROJ/train_lora.py \
  --model_name_or_path "$MODEL" \
  --train_file  $PROJ/data/train.jsonl \
  --eval_file   $PROJ/data/dev.jsonl \
  --output_dir  "$OUT" \
  --bf16 True \
  --attn_implementation sdpa \
  --max_seq_length 512 \
  --packing False \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --logging_steps 10 \
  --eval_strategy steps --eval_steps 100 \
  --save_strategy steps --save_steps 200 --save_total_limit 3 \
  --seed 42 \
  --report_to none

echo "[$(date)] training done, merging adapter"
$PY $PROJ/merge_lora.py --base "$MODEL" --adapter "$OUT" --out "$OUT/merged"

echo "[$(date)] JOB COMPLETE -> $OUT"
