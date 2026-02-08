"""BDD100K PyTorch Dataset for object detection."""
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BDDDataset(Dataset):
    """
    BDD100K Dataset for object detection.
    
    Args:
        root_dir: Root directory containing images and labels
        split: Dataset split ('train', 'val', or 'test')
        transform: Optional transform to be applied on both image and target (bboxes)
        min_bbox_area: Minimum bounding box area to include (filters noise)
    """
    
    # BDD100K has 10 classes for object detection
    CLASSES = [
        'person', 'rider', 'car', 'bus', 'truck',
        'bike', 'motor', 'traffic light', 'traffic sign', 'train'
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        min_bbox_area: int = 16
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.min_bbox_area = min_bbox_area
        
        # Paths
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_file = os.path.join(root_dir, 'labels', f'bdd100k_labels_images_{split}.json')
        
        # Load annotations
        self.images_with_annotations = self._load_images_with_annotations()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
    def _load_images_with_annotations(self) -> List[Dict]:
        """Load BDD100K JSON annotations."""
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r') as f:
            data = json.load(f)
        
        # Filter out images without labels
        valid_annos = []
        for item in data:
            if 'labels' in item and item['labels']:
                # Check if there are any bounding boxes
                has_bbox = any('box2d' in label for label in item['labels'] if label.get('box2d'))
                if has_bbox:
                    valid_annos.append(item)
        
        return valid_annos
    
    def __len__(self) -> int:
        return len(self.images_with_annotations)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Get item at index.
        
        Returns:
            image: PIL Image or tensor (if transform applied)
            target: Dict containing 'boxes', 'labels', 'image_id', etc.
        """
        image_with_annotations = self.images_with_annotations[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_with_annotations['name'])
        image = Image.open(image_path).convert('RGB')
        
        # Parse annotations
        boxes = []
        labels = []
        
        for label in image_with_annotations.get('labels', []):
            if 'box2d' not in label:
                continue
            
            box2d = label['box2d']
            category = label.get('category', '')
            
            if category not in self.class_to_idx:
                continue
            
            # BDD format: x1, y1, x2, y2
            x1 = box2d['x1']
            y1 = box2d['y1']
            x2 = box2d['x2']
            y2 = box2d['y2']
            
            # Filter small boxes
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if area >= self.min_bbox_area and width > 0 and height > 0:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[category])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target
