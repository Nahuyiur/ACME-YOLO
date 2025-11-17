# YOLOv10 Training Guide

This guide explains how to use the unified training script for YOLOv10 with custom improvements.

## Overview

The `train.py` script supports training three different model configurations:
- **MultiScaleFusion**: YOLOv10 with MultiScaleFusion improvement
- **AdaptiveUpSample**: YOLOv10 with AdaptiveUpSample improvement  
- **Combination**: YOLOv10 with all three improvements (AdaptiveUpSample + iAFF + MultiScaleFusion)

## Usage

### Basic Usage

```bash
# Train MultiScaleFusion model from scratch
python train.py --model multiscale_fusion

# Train AdaptiveUpSample model from scratch
python train.py --model adaptive_upsample

# Train combination model from scratch
python train.py --model combination
```

### With Pre-trained Weights

```bash
# Train MultiScaleFusion model with pre-trained weights
python train.py --model multiscale_fusion --pretrained

# Train AdaptiveUpSample model with pre-trained weights
python train.py --model adaptive_upsample --pretrained

# Train combination model with pre-trained weights
python train.py --model combination --pretrained
```

### Custom Parameters

```bash
# Custom training parameters
python train.py --model combination --pretrained --epochs 500 --batch 32 --imgsz 640
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | 'sppelan' | Model type: 'sppelan', 'dysample', or 'combination' |
| `--pretrained` | flag | False | Load pre-trained weights |
| `--epochs` | int | 300 | Number of training epochs |
| `--batch` | int | 16 | Batch size |
| `--imgsz` | int | 640 | Image size |
| `--workers` | int | 8 | Number of workers |
| `--project` | str | 'runs/V10train' | Project directory |
| `--name` | str | 'exp' | Experiment name |
| `--data` | str | 'data/data.yaml' | Dataset configuration file |
| `--pretrained_weights` | str | 'yolov10n.pt' | Pre-trained weights file |

## Examples

### Example 1: Quick Training
```bash
# Train combination model with default settings
python train.py --model combination
```

### Example 2: Fine-tuning
```bash
# Fine-tune combination model with pre-trained weights
python train.py --model combination --pretrained --epochs 100 --batch 8
```

### Example 3: Custom Configuration
```bash
# Custom training with specific parameters
python train.py \
    --model combination \
    --pretrained \
    --epochs 500 \
    --batch 32 \
    --imgsz 640 \
    --workers 16 \
    --project runs/custom_experiment \
    --name my_experiment
```

## Model Configurations

### MultiScaleFusion Model
- Configuration: `ultralytics/cfg/models/v10/yolov10n(MultiScaleFusion).yaml`
- Features: Spatial Pyramid Pooling ELAN module
- Use case: Enhanced feature extraction

### AdaptiveUpSample Model  
- Configuration: `ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample).yaml`
- Features: Dynamic sampling module
- Use case: Improved feature sampling

### Combination Model
- Configuration: `ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample+iAFF+MultiScaleFusion).yaml`
- Features: All three improvements combined
- Use case: Maximum performance (recommended)

## Tips

1. **For best results**: Use the combination model with pre-trained weights
2. **Memory issues**: Reduce batch size if you encounter out-of-memory errors
3. **Training time**: Use more workers if you have multiple GPUs
4. **Dataset**: Make sure your dataset is properly configured in `data/data.yaml`

## Output

Training results will be saved in the specified project directory with the following structure:
```
runs/V10train/exp/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── ...
```
