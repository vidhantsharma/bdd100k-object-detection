"""Global constants for BDD100K object detection."""

# BDD100K object detection classes (in order)
BDD_CLASSES = [
    'person', 
    'rider', 
    'car', 
    'bus', 
    'truck',
    'bike', 
    'motor', 
    'traffic light', 
    'traffic sign', 
    'train'
]

NUM_CLASSES = len(BDD_CLASSES)

# Class ID to name mapping
CLASS_ID_TO_NAME = {idx: name for idx, name in enumerate(BDD_CLASSES)}

# Class name to ID mapping
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(BDD_CLASSES)}

# BDD100K image dimensions
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Dataset splits
SPLITS = ['train', 'val', 'test']
