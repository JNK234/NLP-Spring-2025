#!/bin/bash
#SBATCH --account=p32368
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00 # Increased time for potential pre-training
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --job-name=OpenBookQA_All
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narasimhajwalapuram2026@u.northwestern.edu

# Parse command-line options
run_q1=false
run_q2a=false # Pre-training for Q2
run_q2b=false # Fine-tuning for Q2
run_q3=false

# Default values for Q1
q1_epochs=5
q1_batch_size=8
q1_lr=2e-5

# Default values for Q2a (Pre-training)
q2a_epochs=25
q2a_batch_size=16
q2a_lr=0.0001
q2a_d_model=512
q2a_n_layers=6 
q2a_heads=8
q2a_dropout=0.7
q2a_train_file="NLP/HW#2/wiki103.train.txt"
q2a_valid_file="NLP/HW#2/wiki103.valid.txt"
q2a_test_file="NLP/HW#2/wiki103.test.txt"
q2a_savename="pretrained_transformer" # Base name for saved pre-trained models

# Default values for Q2b (Fine-tuning) - Updated based on tuning
q2b_epochs=20 # Updated default
q2b_batch_size=16 # Updated default
q2b_lr=2e-5 # Updated default
q2b_weight_decay=0.01 # Added default
# This will be dynamically set if Q2a is run
q2b_model_path_default="NLP/HW#2/Logs/Q2a_pretraining_20250512_095731/pretrained_transformer_epoch_7.pth" # Keep a default, but will likely be overridden
# Example path if Q2a is run: /home/wnn7240/JNK/NLP-Spring-2025/NLP/HW#2/Logs/Q2a_pretraining_20250512_095731/pretrained_transformer_epoch_1.pth
q2b_model_path=${q2b_model_path_default}

# Default values for Q3
q3_epochs=60
q3_batch_size=16
q3_lr=5e-6
# This will also use the model from q2a or the default/specified path
q3_model_path_default=${q2b_model_path_default}
q3_model_path=${q3_model_path_default}


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      run_q1=true
      run_q2a=true # Default to run pre-training if --all is specified
      run_q2b=true
      run_q3=true
      shift
      ;;
    --q1) run_q1=true; shift ;;
    --q2a) run_q2a=true; shift ;;
    --q2b) run_q2b=true; shift ;;
    --q2) run_q2b=true; shift ;; # Alias for --q2b if pretraining is skipped
    --q3) run_q3=true; shift ;;
    
    --q1-epochs) q1_epochs="$2"; shift 2 ;;
    --q1-batch-size) q1_batch_size="$2"; shift 2 ;;
    --q1-lr) q1_lr="$2"; shift 2 ;;

    --q2a-epochs) q2a_epochs="$2"; shift 2 ;;
    --q2a-batch-size) q2a_batch_size="$2"; shift 2 ;;
    --q2a-lr) q2a_lr="$2"; shift 2 ;;
    --q2a-train-file) q2a_train_file="$2"; shift 2 ;;
    --q2a-valid-file) q2a_valid_file="$2"; shift 2 ;;
    --q2a-test-file) q2a_test_file="$2"; shift 2 ;;
    
    --q2b-epochs) q2b_epochs="$2"; shift 2 ;;
    --q2b-batch-size) q2b_batch_size="$2"; shift 2 ;;
    --q2b-lr) q2b_lr="$2"; shift 2 ;;
    --q2b-weight-decay) q2b_weight_decay="$2"; shift 2 ;; # Added parsing

    --q3-epochs) q3_epochs="$2"; shift 2 ;;
    --q3-batch-size) q3_batch_size="$2"; shift 2 ;;
    --q3-lr) q3_lr="$2"; shift 2 ;;
    
    --model-path) # This will set the model path for Q2b and Q3 if Q2a is not run
      q2b_model_path="$2"
      q3_model_path="$2"
      shift 2
      ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# If no specific tasks are selected, run all (including q2a by default)
if [[ "$run_q1" == "false" && "$run_q2a" == "false" && "$run_q2b" == "false" && "$run_q3" == "false" ]]; then
  run_q1=true
  run_q2a=true 
  run_q2b=true
  run_q3=true
fi

# Create base logs directory 
LOGS_DIR="/home/wnn7240/JNK/NLP-Spring-2025/NLP/HW#2/Logs"
mkdir -p ${LOGS_DIR}

# Set up environment
module purge
source /home/wnn7240/miniconda/etc/profile.d/conda.sh
conda activate base

# Set environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1

# Install required packages
pip install transformers tqdm matplotlib nltk rouge_score bert_score -q

# Change to working directory
cd /home/wnn7240/JNK/NLP-Spring-2025/

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True)"

echo "Starting OpenBookQA tasks..."
start_time=$(date +%s)

# Run Question 2a: Pre-training
Q2a_DIR="" # Initialize
if [[ "$run_q2a" == "true" ]]; then
  echo "========================================="
  echo "Starting Q2a: Transformer Pre-training"
  echo "========================================="
  
  Q2a_DIR="${LOGS_DIR}/Q2a_pretraining_$(date +%Y%m%d_%H%M%S)"
  mkdir -p ${Q2a_DIR}
  
  python3 NLP/HW#2/q2_pretraining.py \
    -epochs ${q2a_epochs} \
    -batchsize ${q2a_batch_size} \
    -lr ${q2a_lr} \
    -d_model ${q2a_d_model} \
    -n_layers ${q2a_n_layers} \
    -heads ${q2a_heads} \
    -dropout ${q2a_dropout} \
    -train_file "${q2a_train_file}" \
    -valid_file "${q2a_valid_file}" \
    -test_file "${q2a_test_file}" \
    -savename "${q2a_savename}" \
    -dir_name ${Q2a_DIR} > ${Q2a_DIR}/output.log 2> ${Q2a_DIR}/error.log
  
  # Dynamically set the model path for Q2b and Q3 if pre-training was run
  # Assumes the pre-training script saves the final model as <savename>_epoch_<epochs>.pth
  trained_model_path="${Q2a_DIR}/${q2a_savename}_epoch_${q2a_epochs}.pth"
  if [ -f "$trained_model_path" ]; then
    q2b_model_path="$trained_model_path"
    q3_model_path="$trained_model_path"
    echo "Using pre-trained model from this run for Q2b and Q3: $trained_model_path"
  else
    echo "Warning: Pre-trained model from Q2a not found at $trained_model_path. Q2b/Q3 will use default/specified path: ${q2b_model_path_default}"
  fi
  echo "Q2a (Pre-training) completed. Results saved to ${Q2a_DIR}"
fi

# Run Question 1: Classification with BERT
if [[ "$run_q1" == "true" ]]; then
  echo "========================================="
  echo "Starting Q1: BERT Classification approach"
  echo "========================================="
  
  Q1_DIR="${LOGS_DIR}/Q1_classification_$(date +%Y%m%d_%H%M%S)"
  mkdir -p ${Q1_DIR}
  
  python3 NLP/HW#2/q1_classification.py \
    --train_file NLP/HW#2/train_complete.jsonl \
    --valid_file NLP/HW#2/dev_complete.jsonl \
    --test_file NLP/HW#2/test_complete.jsonl \
    --output_dir ${Q1_DIR} \
    --epochs ${q1_epochs} \
    --batch_size ${q1_batch_size} \
    --lr ${q1_lr} > ${Q1_DIR}/output.log 2> ${Q1_DIR}/error.log
    
  echo "Q1 completed. Results saved to ${Q1_DIR}"
fi

# Run Question 2b: Generative approach (Multiple Choice Fine-tuning)
if [[ "$run_q2b" == "true" ]]; then
  echo "========================================="
  echo "Starting Q2b: Generative approach (Multiple Choice Fine-tuning)"
  echo "Using model: ${q2b_model_path}"
  echo "========================================="
  
  Q2b_DIR="${LOGS_DIR}/Q2b_finetuning_$(date +%Y%m%d_%H%M%S)"
  mkdir -p ${Q2b_DIR}
  
  python3 NLP/HW#2/q2_finetune.py \
    --train_file NLP/HW#2/train_complete.jsonl \
    --valid_file NLP/HW#2/dev_complete.jsonl \
    --test_file NLP/HW#2/test_complete.jsonl \
    --model_path "${q2b_model_path}" \
    --output_dir ${Q2b_DIR} \
    --epochs ${q2b_epochs} \
    --batch_size ${q2b_batch_size} \
    --lr ${q2b_lr} \
    --weight_decay ${q2b_weight_decay} \
    --d_model ${q2a_d_model} \
    --n_layers ${q2a_n_layers} \
    --heads ${q2a_heads} \
    --dropout ${q2a_dropout} \
    --clip_grad 1.0 \
    --use_scheduler > ${Q2b_DIR}/output.log 2> ${Q2b_DIR}/error.log

  echo "Q2b (Fine-tuning) completed. Results saved to ${Q2b_DIR}"
fi

# Run Question 3: Generative approach (Full Text Generation)
if [[ "$run_q3" == "true" ]]; then
  echo "========================================="
  echo "Starting Q3: Generative approach (Full Text Generation)"
  echo "Using model: ${q3_model_path}"
  echo "========================================="
  
  Q3_DIR="${LOGS_DIR}/Q3_text_generation_$(date +%Y%m%d_%H%M%S)"
  mkdir -p ${Q3_DIR}
  
  python3 NLP/HW#2/q3_generative_answer.py \
    --train_file NLP/HW#2/train_complete.jsonl \
    --valid_file NLP/HW#2/dev_complete.jsonl \
    --test_file NLP/HW#2/test_complete.jsonl \
    --model_path "${q3_model_path}" \
    --output_dir ${Q3_DIR} \
    --epochs ${q3_epochs} \
    --batch_size ${q3_batch_size} \
    --lr ${q3_lr} \
    --d_model ${q2a_d_model} \
    --n_layers ${q2a_n_layers} \
    --heads ${q2a_heads} \
    --dropout ${q2a_dropout} \
    --clip_grad 1.0 \
    --compare_gpt2 \
    --max_gen_length 50 > ${Q3_DIR}/output.log 2> ${Q3_DIR}/error.log
    
  echo "Q3 completed. Results saved to ${Q3_DIR}"
fi

end_time=$(date +%s)
execution_time=$((end_time - start_time))
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60 ))
seconds=$((execution_time % 60))

echo "========================================="
echo "Execution completed!"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "========================================="

# Usage instructions (for reference):
# To run all questions (including Q2 pre-training):
#   sbatch run_all.sh --all
# To run Q1, Q2 fine-tuning (q2b) and Q3, using a specific pre-trained model:
#   sbatch run_all.sh --q1 --q2b --q3 --model-path /path/to/your/pretrained_model.pth
# To run only Q2 pre-training (q2a):
#   sbatch run_all.sh --q2a
# To run Q2 pre-training (q2a) and then Q2 fine-tuning (q2b) using the just-trained model:
#   sbatch run_all.sh --q2a --q2b
# To specify epochs for Q2 pre-training:
#   sbatch run_all.sh --q2a --q2a-epochs 10
