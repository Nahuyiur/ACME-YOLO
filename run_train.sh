#!/bin/bash
#SBATCH -o train_combination.%j.log
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J train_combination
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_HOME="path/to/cuda"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH


source ~/.bashrc
conda activate yolo_env

nvidia-smi


# Example 1: baseline YOLOv10n training (default dataset & optimizer)
# python train.py --model baseline --epochs 300 --batch 16 --name yolov10n_baseline

# Example 2: CCIRES backbone variant trained from scratch
# python train.py --model ccires --epochs 300 --batch 16 --name yolov10n_ccires

# Example 3: CCIRES backbone with MultiScaleFusion, resume from pretrained weights
# python train.py --model ccires_multiscale --epochs 300 --batch 16 --name yolov10n_ccires_ms \
#   --pretrained --pretrained_weights runs/V10train/exp_ccires/weights/best.pt

# Example 4: Adaptive UpSample + CCIRES + MultiScaleFusion full stack
python train.py --model adaptive_upsample_ccires_multiscale --epochs 300 --batch 16 --name yolov10n_fullstack

# Example 5: Custom YAML path (provide when using --model custom)
# python train.py --model custom --model_config ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample+iAFF+MultiScaleFusion).yaml \
#   --epochs 300 --batch 16 --name yolov10n_custom