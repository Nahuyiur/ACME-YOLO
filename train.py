from ultralytics import YOLOv10
import warnings
import argparse
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='ACME-YOLO training entry point with configurable model variants')
    parser.add_argument('--model', type=str, default='baseline',
                       choices=[
                           'baseline',
                           'multiscale_fusion',
                           'adaptive_upsample',
                           'adaptive_upsample_multiscale',
                           'adaptive_upsample_iaff_multiscale',
                           'ccires',
                           'ccires_multiscale',
                           'adaptive_upsample_ccires',
                           'adaptive_upsample_ccires_multiscale',
                           'custom'
                       ],
                       help='Predefined model variants. Use "custom" when providing --model_config manually.')
    parser.add_argument('--model_config', type=str, default=None,
                       help='Path to model configuration YAML file (required when --model custom)')
    parser.add_argument('--pretrained', action='store_true', 
                       help='Load pre-trained weights (default: False)')
    parser.add_argument('--epochs', type=int, default=300, 
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of workers')
    parser.add_argument('--project', type=str, default='runs/V10train', 
                       help='Project directory')
    parser.add_argument('--name', type=str, default='exp', 
                       help='Experiment name')
    parser.add_argument('--data', type=str, default='data/Visdrone2019_dataset.yaml',
                       help='Dataset configuration file')
    parser.add_argument('--pretrained_weights', type=str, default='yolov10n.pt', 
                       help='Pre-trained weights file')
    
    args = parser.parse_args()
    
    # Model configuration mapping (used only if --model_config is not provided)
    model_configs = {
        'baseline': 'ultralytics/cfg/models/v10/yolov10n.yaml',
        'multiscale_fusion': 'ultralytics/cfg/models/v10/yolov10n(MultiScaleFusion).yaml',
        'adaptive_upsample': 'ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample).yaml',
        'adaptive_upsample_multiscale': 'ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample+MultiScaleFusion).yaml',
        'adaptive_upsample_iaff_multiscale': 'ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample+iAFF+MultiScaleFusion).yaml',
        'ccires': 'ultralytics/cfg/models/v10/yolov10n(CCIRES).yaml',
        'ccires_multiscale': 'ultralytics/cfg/models/v10/yolov10n(CCIRES+MultiScaleFusion).yaml',
        'adaptive_upsample_ccires': 'ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample+CCIRES).yaml',
        'adaptive_upsample_ccires_multiscale': 'ultralytics/cfg/models/v10/yolov10n(AdaptiveUpSample+CCIRES+MultiScaleFusion).yaml'
    }
    
    # Determine which model configuration to load
    if args.model == 'custom':
        if not args.model_config:
            raise ValueError('When --model is "custom", you must provide --model_config.')
        model_yaml_path = args.model_config
        print(f'Using user-specified model configuration: {model_yaml_path}')
    elif args.model_config:
        model_yaml_path = args.model_config
        print(f'--model_config overrides --model. Using: {model_yaml_path}')
    else:
        model_yaml_path = model_configs[args.model]
        print(f'Using predefined model configuration alias: {args.model}')
    
    print('Training configuration:')
    print(f'  Model alias: {args.model}')
    print(f'  Model config: {model_yaml_path}')
    print(f'  Dataset: {args.data}')
    print(f'  Pre-trained: {args.pretrained}')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch}')
    print(f'  Image size: {args.imgsz}')
    print('-' * 50)
    
    # Initialize model
    if args.pretrained:
        print(f'Loading pre-trained model: {args.pretrained_weights}')
        model = YOLOv10(model_yaml_path).load(args.pretrained_weights)
    else:
        print('Training from scratch (no pre-trained weights)')
        model = YOLOv10(model_yaml_path)
    
    # Train model
    print("Starting training...")
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        optimizer='SGD',  # using SGD
        amp=False,  # Turn off amp if training loss becomes NaN
        project=args.project,
        name=args.name,
    )
    
    print("Training completed!")
    return results

if __name__ == '__main__':
    main()