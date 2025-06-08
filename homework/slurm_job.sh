#!/bin/bash
#SBATCH --job-name=hw3
#SBATCH --account=b1229
#SBATCH --partition=b1229
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=ruchibommaraju2026@u.northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

# === Don't activate conda, just use the right Python directly ===
echo "Using Python from: /home/kqm0007/northwestern/conda/envs/hw/bin/python"
echo "Torch version:"
/home/kqm0007/northwestern/conda/envs/hw/bin/python -c "import torch; print(torch.__version__)"

# === Run All 4 Modes ===
for mode in full adapter lora prefix; do
    echo "=== Running mode: $mode ==="
    /home/kqm0007/northwestern/conda/envs/hw/bin/python \
      /home/kqm0007/northwestern/NLP-Spring-2025/homework/train.py \
      --mode "$mode" \
      --epochs 8 \
      --batch_size 8 \
      --lr 1e-4 \
      --dropout 0.2 \
      --log_dir logs/ \
      --model_dir models/
done
