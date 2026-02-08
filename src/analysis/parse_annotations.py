"""Helpers to parse BDD JSON annotations into internal formats."""
import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def parse_bdd_image(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single BDD image with its annotations (labels/objects).
    
    Args:
        image_data: BDD JSON dict for one image containing:
                   - 'name': image filename
                   - 'labels': list of objects (each with box2d, category, etc.)
                   - 'attributes': image-level attributes (weather, time, scene)
    
    Returns:
        Dict representing one image with all its objects:
        {
            'name': image filename,
            'attributes': image-level attributes,
            'boxes': list of [x1,y1,x2,y2] for each object,
            'labels': list of categories for each object,
            'areas': list of areas for each object,
            'occlusions': list of occlusion flags,
            'truncations': list of truncation flags
        }
    """
    result = {
        'name': image_data.get('name', ''),
        'attributes': image_data.get('attributes', {}),
        'boxes': [],
        'labels': [],
        'areas': [],
        'occlusions': [],
        'truncations': []
    }
    
    for label in image_data.get('labels', []):
        if 'box2d' not in label:
            continue
        
        box2d = label['box2d']
        x1, y1 = box2d['x1'], box2d['y1']
        x2, y2 = box2d['x2'], box2d['y2']
        
        # Ensure coordinates are in correct order (handle malformed boxes)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area > 0:
            result['boxes'].append([x1, y1, x2, y2])
            result['labels'].append(label.get('category', 'unknown'))
            result['areas'].append(area)
            result['occlusions'].append(label.get('attributes', {}).get('occluded', False))
            result['truncations'].append(label.get('attributes', {}).get('truncated', False))
    
    return result


def load_bdd_images(json_path: str) -> List[Dict[str, Any]]:
    """
    Load BDD100K images with their annotations from JSON file.
    
    Args:
        json_path: Path to BDD JSON labels file (e.g., bdd100k_labels_images_train.json)
    
    Returns:
        List of dicts, where each dict represents ONE IMAGE with all its objects:
        [
            {
                'name': 'img1.jpg',
                'boxes': [[x1,y1,x2,y2], ...],    # All boxes in this image
                'labels': ['car', 'person', ...],  # All categories in this image
                'areas': [area1, area2, ...],
                'occlusions': [True, False, ...],
                'truncations': [False, True, ...]
            },
            ...
        ]
        
        Note: Only includes images that have at least one bounding box
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    for item in data:
        parsed_image = parse_bdd_image(item)
        if parsed_image['boxes']:  # Only include images with boxes
            images.append(parsed_image)
    
    return images


def get_dataset_statistics(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics from BDD100K dataset.
    
    Args:
        images: List of image dicts (each containing boxes, labels, etc.)
                Each element represents ONE IMAGE with all its objects
    
    Returns:
        Dictionary with dataset-wide statistics:
        {
            'total_images': number of images,
            'total_objects': total number of objects across all images,
            'avg_objects_per_image': average objects per image,
            'category_counts': {category: count},
            'category_stats': {category: {mean_area, median_area, ...}},
            'weather_distribution': {weather: count},
            'timeofday_distribution': {timeofday: count},
            'scene_distribution': {scene: count},
            'occlusion_rate': fraction of occluded objects,
            'truncation_rate': fraction of truncated objects
        }
    """
    total_images = len(images)
    total_objects = sum(len(img['boxes']) for img in images)
    
    # Count by category
    category_counts = defaultdict(int)
    category_areas = defaultdict(list)
    
    for img in images:
        for label, area in zip(img['labels'], img['areas']):
            category_counts[label] += 1
            category_areas[label].append(area)
    
    # Compute area statistics per category
    category_stats = {}
    for category, areas in category_areas.items():
        category_stats[category] = {
            'count': category_counts[category],
            'mean_area': np.mean(areas),
            'median_area': np.median(areas),
            'q1_area': np.percentile(areas, 25),
            'q3_area': np.percentile(areas, 75),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'std_area': np.std(areas)
        }
    
    # Objects per image statistics
    objects_per_image = [len(img['boxes']) for img in images]
    
    # Weather and time attributes (if available)
    weather_counts = defaultdict(int)
    timeofday_counts = defaultdict(int)
    scene_counts = defaultdict(int)
    
    for img in images:
        attrs = img.get('attributes', {})
        if 'weather' in attrs:
            weather_counts[attrs['weather']] += 1
        if 'timeofday' in attrs:
            timeofday_counts[attrs['timeofday']] += 1
        if 'scene' in attrs:
            scene_counts[attrs['scene']] += 1
    
    # Occlusion and truncation statistics
    total_occluded = sum(sum(img['occlusions']) for img in images)
    total_truncated = sum(sum(img['truncations']) for img in images)
    
    stats = {
        'total_images': total_images,
        'total_objects': total_objects,
        'avg_objects_per_image': np.mean(objects_per_image),
        'median_objects_per_image': np.median(objects_per_image),
        'max_objects_per_image': np.max(objects_per_image),
        'min_objects_per_image': np.min(objects_per_image),
        'category_counts': dict(category_counts),
        'category_stats': category_stats,
        'weather_distribution': dict(weather_counts),
        'timeofday_distribution': dict(timeofday_counts),
        'scene_distribution': dict(scene_counts),
        'occlusion_rate': total_occluded / total_objects if total_objects > 0 else 0,
        'truncation_rate': total_truncated / total_objects if total_objects > 0 else 0
    }
    
    return stats
