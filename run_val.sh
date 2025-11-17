#!/bin/bash
#SBATCH -o yolo_val.%j.log
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH -J yolo_val
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_HOME="path/to/cuda"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate yolo_env

nvidia-smi

# Example 1: Validate CCIRES weights on the validation split and write PrettyTables
python val.py --weights runs/V10train/exp_CCIRES/weights/best.pt \
  --split val --batch 16 --name val_ccires --project runs/val --save-paper-metrics

# Example 2: Validate baseline YOLOv10n weights (uncomment to use)
# python val.py --weights runs/V10train/exp_yolov10n/weights/best.pt \
#   --split val --batch 16 --name val_baseline --project runs/val

# Example 3: Validate full-stack AdaptiveUpSample+CCIRES+MultiScaleFusion weights
# python val.py --weights runs/V10train/exp_yolov10n_fullstack/weights/best.pt \
#   --split val --batch 16 --name val_fullstack --project runs/val --save-paper-metrics
