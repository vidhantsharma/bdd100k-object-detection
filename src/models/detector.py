"""Object detection model wrapper for BDD100K."""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Optional, Tuple

from src.utils.constants import BDD_CLASSES, NUM_CLASSES


class BDDDetector(nn.Module):
    """
    Faster R-CNN detector for BDD100K object detection.
    """
    
    # Use centralized class definitions
    NUM_CLASSES = NUM_CLASSES
    CLASS_NAMES = BDD_CLASSES
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs
    ):
        """
        Initialize BDD detector.
        
        Args:
            num_classes: Number of classes (default: 10 for BDD100K)
            pretrained: Use pretrained weights for full model
            pretrained_backbone: Use pretrained backbone
            trainable_backbone_layers: Number of trainable backbone layers (0-5)
                                      When pretrained=True: 3 is typical (fine-tune top layers)
                                      When pretrained=False: 5 recommended (train all layers)
            min_size: Minimum size for image resizing
            max_size: Maximum size for image resizing
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.class_names = self.CLASS_NAMES[:num_classes]
        
        if not pretrained and trainable_backbone_layers < 5:
            print(f"Warning: Training from scratch with only {trainable_backbone_layers} trainable backbone layers.")
            print("Consider setting trainable_backbone_layers=5 to train entire backbone.")
        
        # Load pretrained Faster R-CNN model
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        else:
            weights = None
        
        self.model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            **kwargs
        )
        
        # Replace the classifier head for BDD100K classes
        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        # +1 for background class
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            num_classes + 1  # +1 for background
        )
        
    def forward(self, images, targets=None):
        """
        Forward pass.
        
        Args:
            images: List of images (each a Tensor)
            targets: List of target dicts (for training)
        """
        return self.model(images, targets)
    
    def predict(
        self,
        images: List[torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions on images.
        
        Args:
            images: List of image tensors
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of prediction dicts with 'boxes', 'labels', 'scores'
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            mask = pred['scores'] >= confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, path)
        print(f"Model saved to: {path}")
    
    def load(self, path: str, device: str = 'cpu'):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {path}")
        return self


def create_model(
    num_classes: int = 10,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> BDDDetector:
    """
    Create a BDD detector model.
    
    Args:
        num_classes: Number of object classes
        pretrained: Use pretrained weights
        trainable_backbone_layers: Number of trainable backbone layers (0-5)
        device: Device to load model on
        **kwargs: Additional arguments for BDDDetector
        
    Returns:
        BDDDetector model
    """
    model = BDDDetector(
        num_classes=num_classes,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs
    )
    model = model.to(device)
    return model
