#!/bin/bash
#SBATCH -o yolo_test.%j.log
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH -J yolo_test
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

# Example 1: Evaluate CCIRES weights on the test split
python test.py --weights runs/V10train/exp_CCIRES/weights/best.pt \
  --split test --batch 16 --name test_ccires --project runs/test --save-paper-metrics

# Example 2: Evaluate baseline YOLOv10n weights (uncomment to use)
# python test.py --weights runs/V10train/exp_yolov10n/weights/best.pt \
#   --split test --batch 16 --name test_baseline --project runs/test

# Example 3: Evaluate custom weights with additional options
# python test.py --weights runs/V10train/exp_custom/weights/best.pt \
#   --data data/Visdrone2019_dataset.yaml --split test --batch 8 --imgsz 640 \
#   --name test_custom --project runs/test