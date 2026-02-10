#!/usr/bin/env python3
"""
Main entry point for BDD100K Object Detection Analysis.
Demonstrates the full pipeline: analysis, preprocessing, training, evaluation.
"""
import argparse
import json
import logging
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.utils.config_loader import load_config


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logger = get_logger(__name__)
    logger.setLevel(getattr(logging, level))
    return logger


def run_analysis(args, logger):
    """Run dataset analysis."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Dataset Analysis")
    logger.info("=" * 60)
    
    from src.analysis.parse_annotations import load_bdd_images, get_dataset_statistics
    from src.analysis.dataset_stats import save_statistics_report
    from src.analysis.create_dashboard import create_dashboard
    from src.analysis.detect_anomalies import AnomalyDetector
    from src.analysis.visualize_samples import visualize_outliers
    
    # Load images with their annotations
    labels_file = os.path.join(args.data_root, 'labels', f'bdd100k_labels_images_{args.split}.json')
    logger.info(f"Loading images from: {labels_file}")
    
    if not os.path.exists(labels_file):
        logger.error(f"Labels file not found: {labels_file}")
        logger.info("Please download BDD100K dataset and place it in data/raw/")
        return False
    
    images = load_bdd_images(labels_file)
    logger.info(f"Loaded {len(images)} annotated images")
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = get_dataset_statistics(images)
    
    # Print summary
    logger.info(f"\nDataset Summary ({args.split} split):")
    logger.info(f"  Total Images: {stats['total_images']}")
    logger.info(f"  Total Objects: {stats['total_objects']}")
    logger.info(f"  Avg Objects/Image: {stats['avg_objects_per_image']:.2f}")
    logger.info(f"  Occlusion Rate: {stats['occlusion_rate']:.2%}")
    logger.info(f"  Truncation Rate: {stats['truncation_rate']:.2%}")
    
    logger.info("\nClass Distribution:")
    for cls, count in sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cls:20s}: {count:6d} ({count/stats['total_objects']*100:5.2f}%)")
    
    # Save detailed report
    output_dir = Path(args.output_dir) / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_statistics_report(stats, output_dir / f'{args.split}_statistics.json')
    logger.info(f"\nDetailed statistics saved to: {output_dir / f'{args.split}_statistics.json'}")
    
    # Detect anomalies and identify outliers
    logger.info("\nIdentifying outlier images...")
    # Load raw annotations for anomaly detection (needs full object dictionaries)
    raw_annotations_path = Path(args.data_root) / 'labels' / f'bdd100k_labels_images_{args.split}.json'
    with open(raw_annotations_path, 'r') as f:
        raw_annotations = json.load(f)
    
    detector = AnomalyDetector(raw_annotations, stats)
    anomaly_results = detector.detect_all(top_k=args.top_k)
    
    # Save anomaly results
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super().default(obj)
    
    with open(output_dir / f'{args.split}_anomalies.json', 'w') as f:
        json.dump(anomaly_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Anomaly analysis saved to: {output_dir / f'{args.split}_anomalies.json'}")
    
    # Visualize top outliers
    top_outliers = anomaly_results.get('top_outliers', {}).get('top_outliers', [])
    if top_outliers:
        logger.info(f"\nGenerating visualization for top {len(top_outliers)} outliers...")
        outlier_viz_path = visualize_outliers(
            Path(args.data_root),
            top_outliers,
            args.split,
            output_path=str(output_dir / 'visualizations' / f'{args.split}_top_outliers.png')
        )
        if outlier_viz_path:
            logger.info(f"Outlier visualization created: {outlier_viz_path}")
    
    # Generate interactive dashboard
    logger.info("\nGenerating interactive dashboard...")
    dashboard_path = create_dashboard(
        Path(args.data_root),
        args.split,
        output_file=str(output_dir / f'{args.split}_dashboard.html')
    )
    logger.info(f"Dashboard created: {dashboard_path}")
    logger.info(f"  → Open in browser: file://{Path(dashboard_path).absolute()}")
    
    # Generate training insights page
    logger.info("\nGenerating training insights page...")
    from src.analysis.create_insights import generate_insights_html
    insights_path = output_dir / 'training_insights.html'
    try:
        generate_insights_html(anomaly_results, str(insights_path))
        logger.info(f"Training insights created: {insights_path}")
        logger.info(f"  → Open in browser: file://{insights_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to generate training insights: {e}")
    
    return True


def run_training(args, logger, config):
    """Run model training."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Model Training")
    logger.info("=" * 60)
    
    import torch
    from src.models.detector import create_model
    from src.dataloaders.bdd_dataloader import get_dataloaders
    from src.training.train import Trainer
    from src.training.optimizer import create_optimizer, create_scheduler
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get training config
    train_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Determine if fine-tuning or training from scratch
    finetune = model_config.get('pretrained', True)
    logger.info(f"\nTraining Mode: {'Fine-tuning pretrained model' if finetune else 'Training from scratch'}")
    
    # Load class weights for imbalanced data (from analysis)
    class_weights = None
    class_weights_dict = None  # For weighted sampling
    if train_config.get('use_class_weights', False):
        logger.info("\nLoading class weights from data analysis...")
        stats_file = Path(args.output_dir) / 'analysis' / 'train_statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Calculate inverse frequency weights
            category_counts = stats['category_counts']
            total_objects = stats['total_objects']
            
            from src.datasets.bdd_dataset import BDDDataset
            weights = []
            weights_dict = {}
            for cls in BDDDataset.CLASSES:
                count = category_counts.get(cls, 1)
                weight = total_objects / (len(BDDDataset.CLASSES) * count)
                weights.append(weight)
                weights_dict[cls] = weight
            
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            class_weights_dict = weights_dict
            logger.info(f"Class weights computed:")
            for cls, weight in zip(BDDDataset.CLASSES, weights):
                logger.info(f"  {cls:20s}: {weight:.4f}")
        else:
            logger.warning(f"Statistics file not found: {stats_file}")
            logger.warning("Run analysis stage first to compute class weights")
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(
        num_classes=model_config.get('num_classes', 10),
        pretrained=finetune,
        trainable_backbone_layers=model_config.get('backbone', {}).get('trainable_layers', 3),
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: Faster R-CNN with ResNet50-FPN backbone")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders with weighted sampling option
    logger.info("\nCreating dataloaders...")
    use_weighted_sampling = train_config.get('use_weighted_sampling', False)
    if use_weighted_sampling and class_weights_dict is None:
        logger.warning("Weighted sampling requested but no class weights available. Disabling weighted sampling.")
        use_weighted_sampling = False
    
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_root,
        train_batch_size=train_config.get('batch_size', 4),
        val_batch_size=train_config.get('batch_size', 4),
        num_workers=train_config.get('num_workers', 4),
        min_bbox_area=train_config.get('min_bbox_area', 16),
        use_weighted_sampling=use_weighted_sampling,
        class_weights=class_weights_dict
    )
    
    logger.info(f"Train dataset: {len(train_loader.dataset)} images")
    logger.info(f"Val dataset: {len(val_loader.dataset)} images")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    if use_weighted_sampling:
        logger.info(f"Using weighted sampling to handle class imbalance")
    
    # Store class weights in model for use in loss computation
    if class_weights is not None:
        model.class_weights = class_weights
    
    # Create optimizer
    logger.info("\nCreating optimizer...")
    optimizer = create_optimizer(
        model,
        optimizer_name=train_config.get('optimizer', 'sgd'),
        learning_rate=train_config.get('learning_rate', 0.005),
        momentum=train_config.get('momentum', 0.9),
        weight_decay=train_config.get('weight_decay', 0.0005)
    )
    logger.info(f"Optimizer: {train_config.get('optimizer', 'sgd')}")
    logger.info(f"Learning rate: {train_config.get('learning_rate', 0.005)}")
    
    # Create scheduler
    scheduler = None
    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('type', 'step') != 'none':
        logger.info(f"Scheduler: {scheduler_config.get('type', 'step')}")
        
        # Build scheduler kwargs based on type
        scheduler_kwargs = {
            'step_size': scheduler_config.get('step_size', 3),
            'gamma': scheduler_config.get('gamma', 0.1),
        }
        
        # Add T_max only for cosine scheduler
        if scheduler_config.get('type', 'step') == 'cosine':
            scheduler_kwargs['T_max'] = train_config.get('num_epochs', 10)
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_name=scheduler_config.get('type', 'step'),
            **scheduler_kwargs
        )
    
    # Create output directory for training
    training_output_dir = Path(args.output_dir) / 'training'
    training_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=str(training_output_dir),
        log_interval=train_config.get('log_interval', 10),
        save_interval=train_config.get('save_interval', 1),
        use_tensorboard=True,
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1)
    )
    
    # Start training
    num_epochs = train_config.get('num_epochs', 10)
    logger.info("\n" + "=" * 60)
    logger.info(f"Starting Training - {num_epochs} epochs")
    logger.info("=" * 60)
    
    history = trainer.train(
        num_epochs=num_epochs,
        scheduler=scheduler
    )
    
    # Save final model
    final_model_path = training_output_dir / 'final_model.pth'
    model.save(str(final_model_path))
    logger.info(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = training_output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    return True


def run_evaluation(args, logger, config):
    """Run model evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Model Evaluation")
    logger.info("=" * 60)
    
    import torch
    from src.models.detector import create_model
    from src.dataloaders.bdd_dataloader import make_dataloader
    from src.datasets.bdd_dataset import BDDDataset
    from src.evaluation.evaluate import evaluate_model, save_results, print_results
    from src.evaluation.visualize import visualize_batch
    from src.utils.constants import COCO_TO_BDD_MAPPING
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Setup output directory
    eval_output_dir = Path(args.output_dir) / 'evaluation'
    vis_dir = eval_output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if evaluating pretrained or trained model
    training_output_dir = Path(args.output_dir) / 'training'
    model_path = training_output_dir / 'best_model.pth'
    
    if model_path.exists() and not args.zero_shot:
        # Evaluate trained model
        logger.info(f"Loading trained model from: {model_path}")
        model = create_model(
            num_classes=len(BDDDataset.CLASSES),
            pretrained=False,
            trainable_backbone_layers=5,
            device=device
        )
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle state dict with 'model.' prefix (saved from Detector wrapper)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('model.') for k in state_dict.keys()):
            # Strip 'model.' prefix from all keys
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        
        model.model.load_state_dict(state_dict)
        use_coco_mapping = False
        logger.info("Loaded trained model")
    else:
        # Evaluate pretrained COCO model
        logger.warning(f"Trained model not found at {model_path}")
        logger.info("Using pretrained COCO model with class mapping")
        model = create_model(
            num_classes=len(BDDDataset.CLASSES),
            pretrained=True,
            keep_coco_head=True,
            device=device
        )
        use_coco_mapping = True
        logger.info("Loaded pretrained COCO model")
    
    model.eval()
    
    # Create validation dataloader
    eval_config = config.get('evaluation', {})
    logger.info(f"\nLoading validation data...")
    val_loader = make_dataloader(
        root_dir=args.data_root,
        split='val',
        batch_size=eval_config.get('batch_size', 4),
        shuffle=False,
        num_workers=eval_config.get('num_workers', 4)
    )
    logger.info(f"Validation dataset: {len(val_loader.dataset)} images")
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    results = evaluate_model(
        model=model,
        dataloader=val_loader,
        class_names=BDDDataset.CLASSES,
        device=device,
        iou_threshold=eval_config.get('iou_threshold', 0.5),
        confidence_threshold=0.0,  # AP sweeps all thresholds, don't filter
        use_coco_mapping=use_coco_mapping
    )
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, eval_output_dir / 'results.json')
    logger.info(f"Results saved to: {eval_output_dir / 'results.json'}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Strategy: Collect more images for per-class visualization diversity
    # Collect images from multiple batches
    all_images = []
    all_targets = []
    all_predictions = []
    
    # Collect many images (50+) for better per-class random sampling
    num_images_to_collect = 50
    for batch_images, batch_targets in val_loader:
        batch_images_gpu = [img.to(device) for img in batch_images]
        
        with torch.no_grad():
            batch_predictions = model(batch_images_gpu)
        
        # Apply COCO→BDD mapping if needed
        if use_coco_mapping:
            mapped_preds = []
            for pred in batch_predictions:
                valid_mask = torch.tensor([
                    label.item() in COCO_TO_BDD_MAPPING 
                    for label in pred['labels']
                ], dtype=torch.bool)
                
                if valid_mask.any():
                    bdd_labels = torch.tensor([
                        COCO_TO_BDD_MAPPING[label.item()]
                        for label in pred['labels'][valid_mask]
                    ], dtype=torch.int64, device=pred['labels'].device)
                    
                    mapped_preds.append({
                        'boxes': pred['boxes'][valid_mask],
                        'labels': bdd_labels,
                        'scores': pred['scores'][valid_mask]
                    })
                else:
                    mapped_preds.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'labels': torch.empty(0, dtype=torch.int64, device=device),
                        'scores': torch.empty(0, device=device)
                    })
            batch_predictions = mapped_preds
        
        all_images.extend(batch_images)
        all_targets.extend(batch_targets)
        all_predictions.extend(batch_predictions)
        
        if len(all_images) >= num_images_to_collect:
            break
    
    # Visualize first 6 images for main batch visualization
    visualize_batch(
        images=all_images[:6],
        predictions=all_predictions[:6],
        targets=all_targets[:6],
        class_names=BDDDataset.CLASSES,
        save_path=vis_dir / 'sample_predictions.png',
        max_images=6,
        confidence_threshold=eval_config.get('confidence_threshold', 0.3)
    )
    
    # Generate per-class visualizations using ALL collected images
    logger.info("\nGenerating per-class visualizations...")
    from src.evaluation.visualize import visualize_per_class
    
    visualize_per_class(
        images=all_images,  # Use all collected images for better diversity
        predictions=all_predictions,
        targets=all_targets,
        class_names=BDDDataset.CLASSES,
        save_dir=vis_dir / 'per_class',
        max_per_class=3,
        confidence_threshold=eval_config.get('confidence_threshold', 0.3)
    )
    
    logger.info(f"Visualizations saved to: {vis_dir}/")
    logger.info("\n✓ Evaluation complete!")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='BDD100K Object Detection Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory of BDD dataset (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (overrides config)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split (only for analysis/evaluation stages)')
    parser.add_argument('--stage', type=str, default='analysis',
                       choices=['analysis', 'train', 'evaluate', 'all'],
                       help='Pipeline stage to run')
    parser.add_argument('--zero-shot', action='store_true',
                       help='Use pretrained COCO model for evaluation (no fine-tuning)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Number of top outliers to identify per class (overrides config, default: 3)')
    parser.add_argument('--log-level', type=str, default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.data_root is None:
        args.data_root = config['data']['root']
    if args.output_dir is None:
        args.output_dir = config['data']['output_dir']
    if args.top_k is None:
        args.top_k = config['analysis']['outlier_detection']['top_k']
    if args.log_level is None:
        args.log_level = config['global']['logging']['level']
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("BDD100K Object Detection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data Root: {args.data_root}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Stage: {args.stage}")
    if args.stage in ['analysis', 'preprocessing']:
        logger.info(f"Split: {args.split}")
    
    # Run stages
    success = True
    
    if args.stage == 'all':
        # Run complete pipeline
        if success:
            logger.info("\n>>> Running Analysis Stage (train split)")
            args.split = 'train'  # Analysis on train data
            success = run_analysis(args, logger) and success
        
        if success:
            logger.info("\n>>> Running Training Stage")
            success = run_training(args, logger, config) and success
        
        if success:
            logger.info("\n>>> Running Evaluation Stage (val split)")
            args.split = 'val'  # Evaluate on validation data
            success = run_evaluation(args, logger, config) and success
    
    elif args.stage == 'analysis':
        success = run_analysis(args, logger)
    
    elif args.stage == 'train':
        success = run_training(args, logger, config)
    
    elif args.stage == 'evaluate':
        success = run_evaluation(args, logger, config)
    
    # Summary
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ Pipeline stage completed successfully!")
    else:
        logger.error("✗ Pipeline stage failed. Check logs above.")
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
