"""Weighted sampler for handling class imbalance in object detection."""
from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import WeightedRandomSampler

from src.datasets.bdd_dataset import BDDDataset


def create_weighted_sampler(dataset: BDDDataset, class_weights: Dict[str, float]) -> WeightedRandomSampler:
    """
    Create a weighted sampler based on image-level class distribution.
    
    Images with rare classes (e.g., 'train') get higher sampling probability,
    while images with only common classes (e.g., 'car') get lower probability.
    
    Strategy:
    - For each image, compute weight as: max(class_weights) for all classes in image
    - This oversamples images containing rare objects
    
    Args:
        dataset: BDDDataset instance
        class_weights: Dict mapping class names to weights (higher = rarer)
        
    Returns:
        WeightedRandomSampler for use in DataLoader
    """
    # Compute per-image weights
    image_weights = []
    
    for idx in range(len(dataset)):
        # Get annotation for this image
        image_anno = dataset.images_with_annotations[idx]
        
        # Collect all classes in this image
        classes_in_image = set()
        for label in image_anno.get('labels', []):
            if 'box2d' in label and 'category' in label:
                category = label['category']
                if category in dataset.CLASSES:
                    classes_in_image.add(category)
        
        # Weight = max class weight in image (prioritize rare classes)
        if classes_in_image:
            image_weight = max(class_weights.get(cls, 1.0) for cls in classes_in_image)
        else:
            image_weight = 1.0
        
        image_weights.append(image_weight)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=image_weights,
        num_samples=len(image_weights),
        replacement=True  # Allow sampling with replacement
    )
    
    return sampler
