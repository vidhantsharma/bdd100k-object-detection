"""Create a small subset of BDD100K dataset for faster training/testing."""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import random
from collections import defaultdict


def create_balanced_subset(
    input_json: str,
    input_images_dir: str,
    output_json: str,
    output_images_dir: str,
    num_images: int = 1000,
    min_objects_per_image: int = 2,
    ensure_all_classes: bool = True,
    seed: int = 42
):
    """
    Create a balanced subset of BDD100K dataset.
    
    Strategy:
    1. Filter images with minimum objects
    2. If ensure_all_classes=True, ensure all 10 classes are represented
    3. Randomly sample remaining images
    4. Copy images and create new JSON
    
    Args:
        input_json: Path to original BDD JSON (e.g., data/raw/labels/bdd100k_labels_images_train.json)
        input_images_dir: Path to original images (e.g., data/raw/images/train)
        output_json: Path to save subset JSON (e.g., data/subset/labels/train.json)
        output_images_dir: Path to save subset images (e.g., data/subset/images/train)
        num_images: Number of images to sample (default: 1000)
        min_objects_per_image: Minimum objects per image to include (default: 2)
        ensure_all_classes: Ensure all 10 classes are represented (default: True)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"Creating subset with {num_images} images...")
    print(f"Input JSON: {input_json}")
    print(f"Output JSON: {output_json}")
    
    # Load original annotations
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Total images in original dataset: {len(data)}")
    
    # Define BDD100K classes
    CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 
               'bike', 'motor', 'traffic light', 'traffic sign', 'train']
    
    # Filter valid images (with bounding boxes and minimum objects)
    valid_images = []
    images_by_class = defaultdict(list)
    
    for item in data:
        # Count boxes and classes
        boxes = []
        classes_in_image = set()
        
        for label in item.get('labels', []):
            if 'box2d' in label:
                category = label.get('category', '')
                if category in CLASSES:
                    boxes.append(label)
                    classes_in_image.add(category)
        
        # Only include images with enough objects
        if len(boxes) >= min_objects_per_image:
            valid_images.append(item)
            
            # Track which classes appear in this image
            for cls in classes_in_image:
                images_by_class[cls].append(item)
    
    print(f"Valid images (>= {min_objects_per_image} objects): {len(valid_images)}")
    
    # Check class distribution
    class_counts = {cls: len(images_by_class[cls]) for cls in CLASSES}
    print("\nClass coverage in valid images:")
    for cls in CLASSES:
        print(f"  {cls:20s}: {class_counts[cls]:6d} images")
    
    # Sample strategy
    selected_images = []
    
    if ensure_all_classes:
        print("\nEnsuring all classes are represented...")
        
        # First, ensure each class has at least some images
        images_per_class = max(num_images // (len(CLASSES) * 2), 10)  # At least 10 per class
        
        used_image_names = set()
        
        # Sample images for each class
        for cls in CLASSES:
            available = [img for img in images_by_class[cls] 
                        if img['name'] not in used_image_names]
            
            if available:
                sample_size = min(images_per_class, len(available))
                sampled = random.sample(available, sample_size)
                selected_images.extend(sampled)
                used_image_names.update(img['name'] for img in sampled)
        
        print(f"Selected {len(selected_images)} images covering all classes")
        
        # Fill remaining quota with random valid images
        remaining_quota = num_images - len(selected_images)
        if remaining_quota > 0:
            remaining_pool = [img for img in valid_images 
                            if img['name'] not in used_image_names]
            
            if remaining_pool:
                additional = min(remaining_quota, len(remaining_pool))
                selected_images.extend(random.sample(remaining_pool, additional))
    else:
        # Simple random sampling
        sample_size = min(num_images, len(valid_images))
        selected_images = random.sample(valid_images, sample_size)
    
    print(f"\nFinal subset: {len(selected_images)} images")
    
    # Verify class distribution in final subset
    final_class_counts = defaultdict(int)
    total_objects = 0
    
    for item in selected_images:
        for label in item.get('labels', []):
            if 'box2d' in label:
                category = label.get('category', '')
                if category in CLASSES:
                    final_class_counts[category] += 1
                    total_objects += 1
    
    print(f"Total objects in subset: {total_objects}")
    print("\nObject distribution in subset:")
    for cls in CLASSES:
        count = final_class_counts[cls]
        pct = 100 * count / total_objects if total_objects > 0 else 0
        print(f"  {cls:20s}: {count:6d} ({pct:5.2f}%)")
    
    # Create output directories
    output_json_path = Path(output_json)
    output_images_path = Path(output_images_dir)
    
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_images_path.mkdir(parents=True, exist_ok=True)
    
    # Save subset JSON
    print(f"\nSaving subset JSON to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(selected_images, f, indent=2)
    
    # Copy images
    print(f"Copying images to {output_images_dir}...")
    input_images_path = Path(input_images_dir)
    
    copied = 0
    skipped = 0
    
    for item in selected_images:
        src = input_images_path / item['name']
        dst = output_images_path / item['name']
        
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Warning: Image not found: {src}")
            skipped += 1
    
    print(f"\nCopied {copied} images")
    if skipped > 0:
        print(f"Skipped {skipped} missing images")
    
    print("\n✓ Subset created successfully!")
    print(f"  JSON: {output_json}")
    print(f"  Images: {output_images_dir}")
    print(f"  Total images: {len(selected_images)}")
    print(f"  Total objects: {total_objects}")


def create_subset_for_training(
    data_root: str = "data/raw",
    output_root: str = "data/subset",
    train_size: int = 1000,
    val_size: int = 200,
    seed: int = 42
):
    """
    Create training and validation subsets from BDD100K.
    
    Args:
        data_root: Root directory of original data (default: data/raw)
        output_root: Root directory for subset (default: data/subset)
        train_size: Number of training images (default: 1000)
        val_size: Number of validation images (default: 200)
        seed: Random seed
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    print("=" * 60)
    print("Creating BDD100K Subset for Training")
    print("=" * 60)
    
    # Create training subset
    print("\n[1/2] Creating TRAINING subset...")
    create_balanced_subset(
        input_json=str(data_root / "labels" / "bdd100k_labels_images_train.json"),
        input_images_dir=str(data_root / "images" / "train"),
        output_json=str(output_root / "labels" / "bdd100k_labels_images_train.json"),
        output_images_dir=str(output_root / "images" / "train"),
        num_images=train_size,
        min_objects_per_image=2,
        ensure_all_classes=True,
        seed=seed
    )
    
    # Create validation subset
    print("\n[2/2] Creating VALIDATION subset...")
    create_balanced_subset(
        input_json=str(data_root / "labels" / "bdd100k_labels_images_val.json"),
        input_images_dir=str(data_root / "images" / "val"),
        output_json=str(output_root / "labels" / "bdd100k_labels_images_val.json"),
        output_images_dir=str(output_root / "images" / "val"),
        num_images=val_size,
        min_objects_per_image=2,
        ensure_all_classes=True,
        seed=seed
    )
    
    print("\n" + "=" * 60)
    print("✓ Subset Creation Complete!")
    print("=" * 60)
    print(f"\nTo use the subset for training, update config.yaml:")
    print(f"  data:")
    print(f"    root: \"{output_root}\"")
    print("\nOr run with: python3 main.py --stage train --data-root {output_root}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create BDD100K subset for training")
    parser.add_argument('--data-root', type=str, default='data/raw',
                       help='Root directory of original data')
    parser.add_argument('--output-root', type=str, default='data/subset',
                       help='Root directory for subset')
    parser.add_argument('--train-size', type=int, default=1000,
                       help='Number of training images')
    parser.add_argument('--val-size', type=int, default=200,
                       help='Number of validation images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    create_subset_for_training(
        data_root=args.data_root,
        output_root=args.output_root,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed
    )
