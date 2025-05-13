#!/bin/bash
#SBATCH --account=p32368
#SBATCH --partition=gengpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --job-name=JNK_Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # More CPUs for data loading
#SBATCH --mail-type=ALL     # Only email on completion or failure
#SBATCH --mail-user=narasimhajwalapuram2026@u.northwestern.edu

# Define project name (change this for each run)
PROJECT_NAME="NLP_H2_Q2_pretraining"

# Create project directories 
LOGS_DIR="/home/wnn7240/JNK/NLP/HW#2/Logs"
PROJECT_DIR="${LOGS_DIR}/${PROJECT_NAME}"
mkdir -p ${PROJECT_DIR}

# Set up environment
module purge
source /home/wnn7240/miniconda/etc/profile.d/conda.sh
conda activate base

# Set environment variables for faster training
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1

# Change to working directory
cd /home/wnn7240/JNK/NLP/HW#2

# Run Python script with proper error handling
python3 starter.py \
    -epochs 20 \
    -dir_name ${PROJECT_DIR} > ${PROJECT_DIR}/output.log 2> ${PROJECT_DIR}/error.log \
    -d_model 512 \
    -n_layers 8 \
    -heads 8 \
    -dropout 0.3 \
    -train_file "wiki103.test.txt" \
    -valid_file "wiki103.valid.txt" \
    -test_file "wiki103.test.txt"

    