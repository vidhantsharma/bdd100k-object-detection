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

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# COCO to BDD class mapping
# Maps COCO class IDs to BDD class IDs for pretrained model evaluation
# Only maps classes with clear semantic correspondence
COCO_TO_BDD_MAPPING = {
    1: 0,   # person → person
    2: 5,   # bicycle → bike
    3: 2,   # car → car
    4: 6,   # motorcycle → motor
    6: 3,   # bus → bus
    7: 9,   # train → train
    8: 4,   # truck → truck
    10: 7,  # traffic light → traffic light
    13: 8,  # stop sign → traffic sign (closest match)
}

# Note: COCO class 'rider' (person on bike/motorcycle) doesn't exist in COCO

# Inverse mapping: BDD to COCO (for reference)
BDD_TO_COCO_MAPPING = {v: k for k, v in COCO_TO_BDD_MAPPING.items()}

