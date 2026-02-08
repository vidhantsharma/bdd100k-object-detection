"""Evaluation metrics for object detection."""
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: Box [x1, y1, x2, y2]
        box2: Box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        
    Returns:
        Average Precision
    """
    # Add sentinel values at the beginning and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Calculate area under PR curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_model(
    model,
    data_loader,
    device: str = 'cuda',
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate object detection model.
    
    Args:
        model: Detection model
        data_loader: Validation data loader
        device: Device to evaluate on
        iou_threshold: IoU threshold for matching
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Filter by confidence
            for pred in predictions:
                mask = pred['scores'] >= confidence_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask].cpu(),
                    'labels': pred['labels'][mask].cpu(),
                    'scores': pred['scores'][mask].cpu()
                }
                all_predictions.append(filtered_pred)
            
            # Store targets
            for target in targets:
                all_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
    
    # Compute mAP
    map_score = compute_map(all_predictions, all_targets, iou_threshold)
    
    return {
        'mAP': map_score,
        'num_images': len(all_predictions)
    }


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mean Average Precision (mAP).
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_threshold: IoU threshold for matching
        
    Returns:
        mAP score
    """
    # Collect all predictions and ground truths by class
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Process predictions
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            class_predictions[label.item()].append({
                'image_id': img_idx,
                'box': box,
                'score': score.item()
            })
        
        # Process targets
        for box, label in zip(target['boxes'], target['labels']):
            class_targets[label.item()].append({
                'image_id': img_idx,
                'box': box,
                'matched': False
            })
    
    # Compute AP for each class
    aps = []
    all_classes = set(list(class_predictions.keys()) + list(class_targets.keys()))
    
    for class_id in all_classes:
        preds = class_predictions.get(class_id, [])
        targets_cls = class_targets.get(class_id, [])
        
        if len(targets_cls) == 0:
            continue
        
        if len(preds) == 0:
            aps.append(0.0)
            continue
        
        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        # Match predictions to targets
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        for pred_idx, pred in enumerate(preds):
            # Find matching target
            best_iou = 0.0
            best_target_idx = -1
            
            for target_idx, target in enumerate(targets_cls):
                if target['image_id'] != pred['image_id']:
                    continue
                if target['matched']:
                    continue
                
                iou = compute_iou(pred['box'], target['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx
            
            # Check if match is valid
            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp[pred_idx] = 1
                targets_cls[best_target_idx]['matched'] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(targets_cls)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    # Compute mAP
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)
