"""Minimal visualization for object detection predictions."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict
import random

def visualize_predictions(image: torch.Tensor, 
                         prediction: Dict[str, torch.Tensor],
                         target: Dict[str, torch.Tensor],
                         class_names: List[str],
                         save_path: Path = None,
                         confidence_threshold: float = 0.05):
    """
    Visualize predictions and ground truth on a single image.
    
    Args:
        image: Image tensor [C, H, W]
        prediction: Dict with 'boxes', 'labels', 'scores'
        target: Dict with 'boxes', 'labels'
        class_names: List of class names
        save_path: Path to save figure
        confidence_threshold: Min confidence to display
    """
    # Convert image to numpy
    img = image.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)  # Ensure [0, 1] range
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw ground truth (green boxes)
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.cpu().numpy()
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                 edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        
        class_name = class_names[label.item()] if label.item() < len(class_names) else f"cls_{label.item()}"
        ax.text(x1, y1-5, f'GT: {class_name}', 
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.5),
               fontsize=8, color='white')
    
    # Draw predictions (red boxes)
    mask = prediction['scores'] >= confidence_threshold
    for box, label, score in zip(prediction['boxes'][mask], 
                                 prediction['labels'][mask], 
                                 prediction['scores'][mask]):
        x1, y1, x2, y2 = box.cpu().numpy()
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        class_name = class_names[label.item()] if label.item() < len(class_names) else f"cls_{label.item()}"
        ax.text(x1, y2+15, f'{class_name}: {score:.2f}', 
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
               fontsize=8, color='white')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch(images: List[torch.Tensor],
                   predictions: List[Dict[str, torch.Tensor]],
                   targets: List[Dict[str, torch.Tensor]],
                   class_names: List[str],
                   save_path: Path,
                   max_images: int = 6,
                   confidence_threshold: float = 0.05):
    """Visualize a batch of images with predictions."""
    n_images = min(len(images), max_images)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx in range(n_images):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax = axes[idx]
        ax.imshow(img)
        
        # Ground truth
        for box, label in zip(targets[idx]['boxes'], targets[idx]['labels']):
            x1, y1, x2, y2 = box.cpu().numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                     edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        # Predictions
        pred = predictions[idx]
        mask = pred['scores'] >= confidence_threshold
        for box, label, score in zip(pred['boxes'][mask], 
                                     pred['labels'][mask], 
                                     pred['scores'][mask]):
            x1, y1, x2, y2 = box.cpu().numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                     edgecolor='red', facecolor='none',
                                     linestyle='--')
            ax.add_patch(rect)
        
        ax.set_title(f'Image {idx+1}: {len(pred["boxes"][mask])} preds, {len(targets[idx]["boxes"])} GT\n[Green=GT, Red=Predictions]', 
                    fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    # Add main title with color legend
    plt.suptitle('Model Predictions vs Ground Truth\nGreen = Ground Truth | Red = Model Predictions', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def visualize_per_class(images, predictions, targets, class_names, save_dir,
                        max_per_class=3, confidence_threshold=0.3):
    """
    Create separate visualizations for each class, showing examples with that class
    
    Args:
        images: List of all image tensors
        predictions: List of all prediction dicts
        targets: List of all target dicts
        class_names: List of class names
        save_dir: Directory to save per-class visualizations
        max_per_class: Max images to show per class
        confidence_threshold: Min confidence for predictions
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Group images by class (based on ground truth)
    class_to_indices = {i: [] for i in range(len(class_names))}
    
    for idx, target in enumerate(targets):
        if 'labels' in target and len(target['labels']) > 0:
            labels = target['labels'].cpu().numpy()
            for label in set(labels):  # Unique labels in this image
                if label < len(class_names):
                    class_to_indices[label].append(idx)
    
    # Create visualization for each class
    print("\nGenerating per-class visualizations:")
    for class_id, class_name in enumerate(class_names):
        indices = class_to_indices[class_id]
        if len(indices) == 0:
            print(f"  {class_name}: No examples found in validation set")
            continue
        
        # Randomly select images (or all if fewer than max_per_class)
        num_to_select = min(len(indices), max_per_class)
        selected_indices = random.sample(indices, num_to_select)
        
        num_images = len(selected_indices)
        fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
        if num_images == 1:
            axes = [axes]
        
        for plot_idx, img_idx in enumerate(selected_indices):
            ax = axes[plot_idx]
            
            # Denormalize image
            img = images[img_idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * std) + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
            
            # Plot ground truth boxes
            target = targets[img_idx]
            if 'boxes' in target and len(target['boxes']) > 0:
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                for box, label in zip(gt_boxes, gt_labels):
                    # Only show boxes for the target class
                    if label == class_id:
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.5,
                                                edgecolor='green', facecolor='none')
                        ax.add_patch(rect)
            
            # Plot predictions
            pred = predictions[img_idx]
            if 'boxes' in pred and len(pred['boxes']) > 0:
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if score >= confidence_threshold and label == class_id:
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                                edgecolor='red', facecolor='none',
                                                linestyle='--')
                        ax.add_patch(rect)
            
            ax.set_title(f'Example {plot_idx+1}\n[Green=GT, Red=Predictions]', fontsize=9)
        
        plt.suptitle(f'Class: {class_name} ({len(indices)} images in val set)\nGreen = Ground Truth | Red = Model Predictions',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / f'{class_name}_examples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  {class_name}: {num_images} examples → {save_path.name}")
    
    print(f"\nPer-class visualizations saved to: {save_dir}")

