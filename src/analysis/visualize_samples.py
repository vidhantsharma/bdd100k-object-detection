"""
Visualize Interesting/Unique Samples from BDD100K Dataset

This script finds and visualizes:
- Rare class examples (train class)
- Heavily occluded scenes
- Dense scenes (many objects)
- Sparse scenes (few objects)
- Challenging conditions (night, bad weather)
- Unusual object sizes
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple
import random


# BDD100K color scheme for each class
CLASS_COLORS = {
    'car': (0, 114, 189),
    'truck': (217, 83, 25),
    'bus': (237, 177, 32),
    'person': (126, 47, 142),
    'rider': (119, 172, 48),
    'bike': (77, 190, 238),
    'motor': (162, 20, 47),
    'traffic light': (255, 87, 51),
    'traffic sign': (0, 255, 255),
    'train': (128, 0, 128)
}


def draw_bbox_on_image(image_path: str, annotations: Dict, save_path: str = None) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image_path: Path to image file
        annotations: Annotation dictionary
        save_path: Optional path to save annotated image
        
    Returns:
        Annotated image as numpy array
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw each bounding box
    for obj in annotations.get('labels', []):
        if 'box2d' not in obj:
            continue
            
        box = obj['box2d']
        category = obj.get('category', 'unknown')
        attributes = obj.get('attributes', {})
        
        # Get color for this class
        color = CLASS_COLORS.get(category, (255, 255, 255))
        
        # Draw rectangle
        x1, y1 = int(box['x1']), int(box['y1'])
        x2, y2 = int(box['x2']), int(box['y2'])
        
        # Thicker line if occluded
        thickness = 3 if attributes.get('occluded', False) else 2
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = category
        if attributes.get('occluded'):
            label += " (OCC)"
        if attributes.get('truncated'):
            label += " (TRN)"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save if requested
    if save_path:
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), save_img)
    
    return img


def visualize_outliers(data_root: Path, outlier_list: List[Dict], split: str = 'train', 
                       output_path: str = None) -> str:
    """
    Visualize top outlier object detections grouped by class
    
    Args:
        data_root: Root directory containing BDD100K data
        outlier_list: List of outlier objects with image_name, object_index, score, reasons
        split: Dataset split
        output_path: Output file path
        
    Returns:
        Path to saved visualization
    """
    if not outlier_list:
        print("No outliers to visualize")
        return None
    
    # Load annotations
    label_file = data_root / 'labels' / f'bdd100k_labels_images_{split}.json'
    with open(label_file) as f:
        all_annotations = json.load(f)
    
    # Create map for quick lookup
    ann_map = {ann['name']: ann for ann in all_annotations}
    
    # Group outliers by class
    outliers_by_class = {}
    for outlier in outlier_list:
        cat = outlier['category']
        if cat not in outliers_by_class:
            outliers_by_class[cat] = []
        
        img_name = outlier['image_name']
        if img_name in ann_map:
            ann = ann_map[img_name].copy()
            ann['outlier_object_idx'] = outlier['object_index']
            ann['outlier_score'] = outlier['outlier_score']
            ann['outlier_reasons'] = outlier['reasons']
            ann['outlier_category'] = outlier['category']
            outliers_by_class[cat].append((img_name, ann, outlier))
    
    # Create visualization with sections for each class
    n_cols = 3
    total_samples = len(outlier_list)
    n_rows_per_class = {}
    total_rows = 0
    
    # Calculate rows needed for each class
    for cat in sorted(outliers_by_class.keys()):
        n_samples = len(outliers_by_class[cat])
        rows_needed = (n_samples + n_cols - 1) // n_cols
        n_rows_per_class[cat] = rows_needed
        total_rows += rows_needed + 1  # +1 for class header
    
    fig = plt.figure(figsize=(18, 6 * total_rows))
    
    # Use GridSpec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(total_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    current_row = 0
    
    for class_idx, cat in enumerate(sorted(outliers_by_class.keys())):
        outlier_samples = outliers_by_class[cat]
        
        # Add class header
        ax_header = fig.add_subplot(gs[current_row, :])
        # Normalize color to 0-1 range for matplotlib
        class_color = CLASS_COLORS.get(cat, (200, 200, 200))
        class_color_normalized = tuple(c / 255.0 for c in class_color)
        ax_header.text(0.5, 0.5, f'━━━ CLASS: {cat.upper()} ({len(outlier_samples)} outliers) ━━━', 
                      ha='center', va='center', fontsize=16, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor=class_color_normalized, 
                               alpha=0.3, edgecolor='black', linewidth=2))
        ax_header.axis('off')
        current_row += 1
        
        # Add outlier images for this class
        for idx, (img_name, ann, outlier) in enumerate(outlier_samples):
            row = current_row + (idx // n_cols)
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Find image file
            img_path = None
            for split_name in [split, 'train', 'val', 'test']:
                test_path = data_root / 'images' / split_name / img_name
                if test_path.exists():
                    img_path = test_path
                    break
            
            if not img_path or not img_path.exists():
                ax.text(0.5, 0.5, f'Image not found:\n{img_name}', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                ax.text(0.5, 0.5, f'Error loading:\n{img_name}', 
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw all boxes (outlier highlighted differently)
            outlier_idx = ann.get('outlier_object_idx', -1)
            outlier_box = None
            
            for obj_idx, obj in enumerate(ann.get('labels', [])):
                if 'box2d' not in obj:
                    continue
                
                box = obj['box2d']
                category = obj.get('category', 'unknown')
                
                # Highlight outlier object
                if obj_idx == outlier_idx:
                    color = (255, 0, 0)  # Bright red for outlier
                    thickness = 4
                    outlier_box = box
                else:
                    color = CLASS_COLORS.get(category, (255, 255, 255))
                    thickness = 1  # Thin for non-outliers
                
                x1, y1 = int(box['x1']), int(box['y1'])
                x2, y2 = int(box['x2']), int(box['y2'])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Label only the outlier object
                if obj_idx == outlier_idx:
                    label = f"{category.upper()}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x1, y1 - label_h - 8), (x1 + label_w + 10, y1), (255, 0, 0), -1)
                    cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ax.imshow(img)
            
            # Title with score
            score = ann.get('outlier_score', 0)
            title = f"Score: {score:.1f} | {img_name}"
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            # Add reasons as text
            reasons = ann.get('outlier_reasons', [])
            if reasons:
                reason_text = '\n'.join(f"• {r}" for r in reasons[:3])  # Top 3 reasons
                ax.text(0.02, 0.98, reason_text, 
                       transform=ax.transAxes,
                       fontsize=7, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            ax.axis('off')
        
        # Update current_row for next class
        current_row += n_rows_per_class[cat]
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_dir = Path('data/analysis/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{split}_top_outliers.png'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Outlier visualization saved: {output_path}")
    return str(output_path)
