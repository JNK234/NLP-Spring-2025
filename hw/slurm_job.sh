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

# === Load Anaconda and Activate Your Conda Environment ===
module purge
module load anaconda3
source /home/kqm0007/northwestern/conda/etc/profile.d/conda.sh
conda activate hw

# === Run All 4 Modes ===
for mode in full adapter lora prefix; do
    echo "=== Running mode: $mode ==="
    python3 /home/kqm0007/northwestern/NLP-Spring-2025/hw/train.py \
      --mode "$mode" \
      --epochs 10 \
      --batch_size 8 \
      --lr 1e-4 \
      --dropout 0.2 \
      --log_dir logs/ \
      --model_dir models/
done
