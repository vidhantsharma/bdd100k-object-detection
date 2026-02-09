"""Minimal evaluation module for object detection."""
import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
from tqdm import tqdm


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using COCO-style 101-point interpolation.
    
    COCO AP samples precision at 101 recall thresholds [0.00:0.01:1.00]
    with interpolated precision values (maximum precision at recall >= threshold).
    """
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # COCO 101-point interpolation: sample at 101 recall thresholds
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 101):
        # Find precisions at recalls >= threshold
        indices = np.where(recalls >= recall_threshold)[0]
        if len(indices) > 0:
            ap += precisions[indices[0]]
    
    return ap / 101


def evaluate_model(model, dataloader, class_names: List[str], device, 
                   iou_threshold: float = 0.5, confidence_threshold: float = 0.0,
                   use_coco_mapping: bool = False):
    """
    Evaluate object detection model with COCO-style AP computation.
    
    Args:
        confidence_threshold: For visualization only. AP sweeps all thresholds (use 0.0).
    
    Returns:
        dict with mAP, per_class metrics, and confusion matrix
    """
    model.eval()
    
    # Collect predictions and targets by class WITH IMAGE IDs
    all_predictions = {i: [] for i in range(len(class_names))}
    all_targets = {i: [] for i in range(len(class_names))}
    
    print("Running inference...")
    image_id = 0  # Track image index
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            # Apply COCO→BDD mapping if enabled
            if use_coco_mapping:
                from src.utils.constants import COCO_TO_BDD_MAPPING
                mapped_preds = []
                for pred in predictions:
                    # Filter predictions that have BDD equivalents
                    valid_mask = torch.tensor([
                        label.item() in COCO_TO_BDD_MAPPING 
                        for label in pred['labels']
                    ], dtype=torch.bool)
                    
                    if valid_mask.any():
                        # Remap COCO IDs to BDD IDs
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
                        # No valid predictions, return empty
                        mapped_preds.append({
                            'boxes': torch.empty((0, 4), device=device),
                            'labels': torch.empty(0, dtype=torch.int64, device=device),
                            'scores': torch.empty(0, device=device)
                        })
                predictions = mapped_preds
            
            # Store predictions and targets with image_id
            for pred, target in zip(predictions, targets):
                for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                    all_predictions[label.item()].append({
                        'box': box.cpu(),
                        'score': score.item(),
                        'image_id': image_id
                    })
                
                for box, label in zip(target['boxes'], target['labels']):
                    all_targets[label.item()].append({
                        'box': box.cpu(),
                        'matched': False,
                        'image_id': image_id
                    })
                
                image_id += 1  # Increment for each image
    
    print(f"\nComputing mAP (IoU={iou_threshold}, COCO-style all-point interpolation)...")
    
    # Compute per-class AP
    class_metrics = {}
    aps = []
    
    for class_id, class_name in enumerate(class_names):
        preds = all_predictions[class_id]
        targets = all_targets[class_id]
        
        if len(preds) == 0 and len(targets) == 0:
            continue
        
        if len(preds) == 0:
            class_metrics[class_name] = {
                'AP': 0.0, 'precision': 0.0, 'recall': 0.0, 'F1': 0.0,
                'num_predictions': 0, 'num_ground_truths': len(targets)
            }
            aps.append(0.0)
            continue
        
        if len(targets) == 0:
            class_metrics[class_name] = {
                'AP': 0.0, 'precision': 0.0, 'recall': 0.0, 'F1': 0.0,
                'num_predictions': len(preds), 'num_ground_truths': 0
            }
            continue
        
        for t in targets:
            t['matched'] = False
        
        # Sort predictions by score (descending)
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        # Match predictions to targets (per-image matching)
        true_positives = []
        false_positives = []
        
        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            
            for idx, target in enumerate(targets):
                if target['matched']:
                    continue
                
                if target['image_id'] != pred['image_id']:
                    continue
                
                iou = compute_iou(pred['box'], target['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold:
                targets[best_idx]['matched'] = True
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)
        
        # Compute precision-recall
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recalls = tp_cumsum / len(targets)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP (COCO-style all-point interpolation)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
        # Final metrics
        precision = precisions[-1] if len(precisions) > 0 else 0.0
        recall = recalls[-1] if len(recalls) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[class_name] = {
            'AP': float(ap),
            'precision': float(precision),
            'recall': float(recall),
            'F1': float(f1),
            'num_predictions': len(preds),
            'num_ground_truths': len(targets)
        }
    
    # Compute mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    
    return {
        'mAP': float(mAP),
        'per_class': class_metrics,
        'num_classes_evaluated': len(aps),
        'iou_threshold': iou_threshold,
        'confidence_threshold': confidence_threshold
    }


def save_results(results: dict, output_path: Path):
    """Save evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "="*60)
    print(f"mAP@{results['iou_threshold']}: {results['mAP']:.4f}")
    print("="*60)
    
    print("\nPer-class results:")
    print(f"{'Class':<15} {'AP':>6} {'P':>6} {'R':>6} {'F1':>6} {'Preds':>6} {'GT':>6}")
    print("-"*60)
    
    for class_name, metrics in sorted(results['per_class'].items(), 
                                     key=lambda x: x[1]['AP'], reverse=True):
        print(f"{class_name:<15} "
              f"{metrics['AP']:>6.3f} "
              f"{metrics['precision']:>6.3f} "
              f"{metrics['recall']:>6.3f} "
              f"{metrics['F1']:>6.3f} "
              f"{metrics['num_predictions']:>6} "
              f"{metrics['num_ground_truths']:>6}")
