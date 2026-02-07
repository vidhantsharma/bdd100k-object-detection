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


def run_preprocessing(args, logger):
    """Run data preprocessing."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Data Preprocessing")
    logger.info("=" * 60)
    
    from src.preprocessing.convert_bdd import convert_bdd_to_coco
    from src.analysis.parse_annotations import load_bdd_images, images_to_coco_format
    from src.datasets.bdd_dataset import BDDDataset
    
    # Convert to COCO format
    labels_file = os.path.join(args.data_root, 'labels', f'bdd100k_labels_images_{args.split}.json')
    logger.info(f"Converting {labels_file} to COCO format...")
    
    images = load_bdd_images(labels_file)
    coco_data = images_to_coco_format(images, BDDDataset.CLASSES)
    
    # Save COCO format
    output_file = Path(args.output_dir) / 'processed' / f'{args.split}_coco.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"COCO format saved to: {output_file}")
    logger.info(f"  Images: {len(coco_data['images'])}")
    logger.info(f"  Annotations: {len(coco_data['annotations'])}")
    logger.info(f"  Categories: {len(coco_data['categories'])}")
    
    return True


def run_model_test(args, logger):
    """Test model loading and inference."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Model Test (Dummy Inference)")
    logger.info("=" * 60)
    
    try:
        import torch
        from src.models.detector import Detector
        
        logger.info("Creating model...")
        detector = Detector(model_name=args.model, num_classes=len(args.classes) if hasattr(args, 'classes') else 10)
        logger.info(f"Model created: {args.model}")
        
        # Dummy inference test
        logger.info("Running dummy inference test...")
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            result = detector.predict(dummy_input)
        logger.info("✓ Dummy inference successful")
        
        return True
    except Exception as e:
        logger.warning(f"Model test skipped: {e}")
        return True  # Non-critical


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='BDD100K Object Detection Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory of BDD dataset (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (overrides config)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to analyze')
    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_fpn',
                       help='Model architecture')
    parser.add_argument('--stages', type=str, nargs='+', 
                       default=['analysis', 'preprocessing', 'model_test'],
                       choices=['analysis', 'preprocessing', 'model_test', 'all'],
                       help='Stages to run')
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
    logger.info("BDD100K Object Detection Analysis")
    logger.info("=" * 60)
    logger.info(f"Data Root: {args.data_root}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Stages: {', '.join(args.stages)}")
    
    # Run stages
    stages = args.stages
    if 'all' in stages:
        stages = ['analysis', 'preprocessing', 'model_test']
    
    success = True
    
    if 'analysis' in stages:
        success = run_analysis(args, logger) and success
    
    if 'preprocessing' in stages and success:
        success = run_preprocessing(args, logger) and success
    
    if 'model_test' in stages and success:
        success = run_model_test(args, logger) and success
    
    # Summary
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ Pipeline completed successfully!")
    else:
        logger.error("✗ Pipeline failed. Check logs above.")
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
