#!/bin/bash
# ---------------------------------------------------------------------------
# GPU READINESS PREFLIGHT  (run this BEFORE any training job)
# Submit with:
#   mkdir -p /homes/01/mxpoudel/Grad/nl2sh/slurm_outputs
#   sbatch /homes/01/mxpoudel/Grad/nl2sh/preflight_gpu.sbatch
# Costs ~2 minutes of an A100. Answers every unknown in one shot.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nl2sh_preflight
#SBATCH --partition=suliaoma
#SBATCH --qos=bxmarg
#SBATCH --nodelist=gpu003
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/homes/01/mxpoudel/Grad/nl2sh/slurm_outputs/preflight_%j.out
#SBATCH --error=/homes/01/mxpoudel/Grad/nl2sh/slurm_outputs/preflight_%j.err

set -x
PIXI=/homes/01/mxpoudel/Programs/gdal/.pixi/envs/default
PY=$PIXI/bin/python

echo "=== HOST / OS ==="
hostname; cat /etc/redhat-release 2>/dev/null; uname -r; ldd --version | head -1

echo "=== NVIDIA DRIVER + GPU MEMORY (settles the 40GB vs 80GB question) ==="
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv

echo "=== PIXI PYTHON RUNS ON ROCKY 9 COMPUTE NODE? ==="
$PY -V
$PY -c "import sys; print(sys.executable); print(sys.version)"

echo "=== TORCH SEES THE A100? ==="
$PY - <<'EOF'
import torch, time
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("name:", p.name)
    print("total_mem_GiB:", round(p.total_memory/1024**3, 1))
    print("capability:", f"{p.major}.{p.minor}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    # real matmul + a bf16 throughput probe (gives you the MFU denominator)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): c = a @ b
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): c = a @ b
    torch.cuda.synchronize()
    dt = (time.time()-t0)/20
    print("bf16 8192^3 matmul TFLOPS:", round(2*8192**3/dt/1e12, 1))
    print("peak_alloc_GiB:", round(torch.cuda.max_memory_allocated()/1024**3, 2))
    # SDPA path (we will train with attn_implementation="sdpa", NOT flash-attn)
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    q = torch.randn(2, 16, 512, 64, device="cuda", dtype=torch.bfloat16)
    print("sdpa ok:", sdpa(q, q, q).shape)
EOF

echo "=== NCCL / 2-GPU sanity (only matters if you later use --gres=gpu:a100:2) ==="
$PY -c "import torch;print('nccl available:', torch.distributed.is_nccl_available())"

echo "=== DOES THE COMPUTE NODE HAVE OUTBOUND INTERNET? (HF download from inside a job) ==="
timeout 20 curl -sI https://huggingface.co | head -1 || echo "NO OUTBOUND HTTPS -> must pre-download models/datasets on the LOGIN node"

echo "=== DISK ==="
df -h /homes/01 | tail -1
du -sh /homes/01/mxpoudel/.cache/huggingface 2>/dev/null

echo "=== TRAINING VENV (only exists after setup_venv.sh has been run) ==="
VENV=/homes/01/mxpoudel/Grad/nl2sh/venv
if [ -x "$VENV/bin/python" ]; then
  $VENV/bin/python - <<'EOF'
import torch, transformers, peft, trl, datasets, accelerate
print("torch      ", torch.__version__)          # MUST still be 2.6.0+cu118
print("transformers", transformers.__version__)
print("peft       ", peft.__version__)
print("trl        ", trl.__version__)
print("datasets   ", datasets.__version__)
print("accelerate ", accelerate.__version__)
assert torch.__version__.startswith("2.6.0+cu118"), "TORCH WAS CLOBBERED - rebuild venv with --no-deps"
assert torch.cuda.is_available(), "venv torch lost CUDA"
print("VENV OK")
EOF
else
  echo "venv not built yet (expected on first run)"
fi

echo "=== PREFLIGHT COMPLETE ==="
