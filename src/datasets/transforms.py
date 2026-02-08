"""Transform and augmentation utilities for BDD dataset."""
import random
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to Tensor."""
    
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            return image, target
        return image


class Normalize:
    """Normalize image with mean and std."""
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is not None:
            return image, target
        return image


class RandomHorizontalFlip:
    """Random horizontal flip with probability p."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = F.hflip(image)
            if target is not None and 'boxes' in target:
                # Get image width (handle both PIL Image and Tensor)
                if isinstance(image, torch.Tensor):
                    _, width = image.shape[-2:]
                else:
                    width, _ = image.size
                
                # Flip bounding boxes
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        if target is not None:
            return image, target
        return image


class RandomResize:
    """Randomly resize image while keeping aspect ratio."""
    
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            w, h = image.size
        
        # Choose random size
        size = random.randint(self.min_size, self.max_size)
        
        # Calculate new dimensions
        if w < h:
            new_w = size
            new_h = int(size * h / w)
        else:
            new_h = size
            new_w = int(size * w / h)
        
        # Resize image
        if isinstance(image, torch.Tensor):
            image = F.resize(image, [new_h, new_w])
        else:
            image = F.resize(image, [new_h, new_w])
        
        # Scale boxes if target provided
        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            boxes[:, 0::2] *= new_w / w  # x coordinates
            boxes[:, 1::2] *= new_h / h  # y coordinates
            target['boxes'] = boxes
        
        if target is not None:
            return image, target
        return image


class ColorJitter:
    """Apply color jittering (brightness, contrast, saturation, hue)."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image, target=None):
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        
        if target is not None:
            return image, target
        return image


def get_train_transform():
    """Get training transforms with augmentation."""
    return Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomResize(min_size=600, max_size=1000),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    """Get validation transforms without augmentation."""
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collate_fn(batch):
    """Custom collate function for dataloader."""
    return tuple(zip(*batch))
