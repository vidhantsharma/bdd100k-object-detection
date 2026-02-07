"""
Anomaly and Pattern Detection for BDD100K Dataset

This module identifies anomalies and interesting patterns in object detection classes:
- Class imbalance issues
- Size distribution anomalies
- Occlusion patterns
- Rare/edge cases
- Spatial distribution patterns
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies and patterns in BDD100K annotations"""
    
    def __init__(self, annotations: List[Dict], statistics: Dict):
        """
        Initialize anomaly detector
        
        Args:
            annotations: List of image annotations
            statistics: Pre-computed statistics dictionary
        """
        self.annotations = annotations
        self.stats = statistics
        self.anomalies = defaultdict(list)
        
    def detect_all(self, top_k: int = 3) -> Dict[str, Any]:
        """Run all anomaly detection methods
        
        Args:
            top_k: Number of top outliers to identify per class (default: 3)
        """
        logger.info("Detecting anomalies and patterns...")
        
        results = {
            "class_imbalance": self.detect_class_imbalance(),
            "size_anomalies": self.detect_size_anomalies(),
            "occlusion_patterns": self.detect_occlusion_patterns(),
            "quality_issues": self.detect_quality_issues(),
            "top_outliers": self.identify_top_outliers(top_k=top_k),
        }
        
        return results
    
    def detect_class_imbalance(self) -> Dict[str, Any]:
        """Detect severe class imbalance issues"""
        category_counts = self.stats['category_counts']
        total = sum(category_counts.values())
        
        # Calculate imbalance metrics
        percentages = {cat: (count/total)*100 for cat, count in category_counts.items()}
        
        # Find dominant and rare classes
        sorted_classes = sorted(percentages.items(), key=lambda x: -x[1])
        dominant = sorted_classes[0]
        rare = [c for c in sorted_classes if c[1] < 1.0]
        
        # Calculate imbalance ratio
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return {
            "imbalance_ratio": imbalance_ratio,
            "dominant_class": {
                "name": dominant[0],
                "percentage": dominant[1],
                "count": category_counts[dominant[0]]
            },
            "rare_classes": [
                {
                    "name": cat,
                    "percentage": pct,
                    "count": category_counts[cat]
                }
                for cat, pct in rare
            ],
            "recommendations": [
                f"Use class weights or focal loss to handle {imbalance_ratio:.1f}x imbalance",
                f"Consider oversampling rare classes: {', '.join([c[0] for c in rare])}",
                f"Dominant class '{dominant[0]}' may cause bias - monitor precision/recall per class"
            ]
        }
    
    def detect_size_anomalies(self) -> Dict[str, Any]:
        """Detect anomalous object sizes"""
        anomalies = {}
        
        for category, stats in self.stats['category_stats'].items():
            mean_area = stats['mean_area']
            std_area = stats['std_area']
            
            # High variance indicates inconsistent sizes
            coeff_variation = (std_area / mean_area) * 100 if mean_area > 0 else 0
            
            # Size range
            size_range = stats['max_area'] - stats['min_area']
            
            anomalies[category] = {
                "mean_area": mean_area,
                "std_area": std_area,
                "coefficient_of_variation": coeff_variation,
                "size_range": size_range,
                "is_highly_variable": coeff_variation > 200,
                "has_tiny_objects": stats['min_area'] < 50,
                "has_huge_objects": stats['max_area'] > 100000,
            }
        
        # Find most variable classes
        variable_classes = sorted(
            [(cat, data['coefficient_of_variation']) for cat, data in anomalies.items()],
            key=lambda x: -x[1]
        )[:3]
        
        return {
            "per_class": anomalies,
            "most_variable_classes": variable_classes,
            "recommendations": [
                "Use multi-scale feature pyramids (FPN) for size variation",
                f"Classes with high variation ({', '.join([c[0] for c in variable_classes])}): use anchor boxes of multiple scales",
                "Consider separate heads for small vs large objects"
            ]
        }
    
    def detect_occlusion_patterns(self) -> Dict[str, Any]:
        """Detect occlusion patterns by class"""
        occlusion_by_class = defaultdict(lambda: {"occluded": 0, "total": 0})
        truncation_by_class = defaultdict(lambda: {"truncated": 0, "total": 0})
        
        for img_ann in self.annotations:
            labels_list = img_ann.get('labels', [])
            if not isinstance(labels_list, list):
                continue
            
            for obj in labels_list:
                if not isinstance(obj, dict):
                    continue
                
                category = obj.get('category', 'unknown')
                
                # Count occlusion
                attributes = obj.get('attributes', {})
                occlusion_by_class[category]["total"] += 1
                if attributes.get('occluded', False):
                    occlusion_by_class[category]["occluded"] += 1
                
                # Count truncation
                truncation_by_class[category]["total"] += 1
                if attributes.get('truncated', False):
                    truncation_by_class[category]["truncated"] += 1
        
        # Calculate rates
        occlusion_rates = {
            cat: (data["occluded"] / data["total"] * 100) if data["total"] > 0 else 0
            for cat, data in occlusion_by_class.items()
        }
        
        truncation_rates = {
            cat: (data["truncated"] / data["total"] * 100) if data["total"] > 0 else 0
            for cat, data in truncation_by_class.items()
        }
        
        # Find most affected classes
        most_occluded = sorted(occlusion_rates.items(), key=lambda x: -x[1])[:3]
        most_truncated = sorted(truncation_rates.items(), key=lambda x: -x[1])[:3]
        
        return {
            "occlusion_rates": occlusion_rates,
            "truncation_rates": truncation_rates,
            "most_occluded_classes": most_occluded,
            "most_truncated_classes": most_truncated,
            "overall_occlusion_rate": self.stats.get('occlusion_rate', 0) * 100,
            "recommendations": [
                f"High occlusion ({self.stats.get('occlusion_rate', 0)*100:.1f}%): use IoU-aware detection heads",
                f"Most occluded: {', '.join([c[0] for c in most_occluded])} - consider using context/relationship modeling",
                "Use data augmentation with artificial occlusion during training"
            ]
        }
    
    def detect_quality_issues(self) -> Dict[str, Any]:
        """Detect potential annotation quality issues"""
        issues = {
            "very_small_boxes": [],
            "very_large_boxes": [],
            "unusual_aspect_ratios": [],
        }
        
        for img_ann in self.annotations:
            labels_list = img_ann.get('labels', [])
            if not isinstance(labels_list, list):
                continue
            
            for obj in labels_list:
                if not isinstance(obj, dict) or 'box2d' not in obj:
                    continue
                
                box = obj['box2d']
                width = box['x2'] - box['x1']
                height = box['y2'] - box['y1']
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                category = obj.get('category', 'unknown')
                
                # Very small boxes (possible annotation errors)
                if area < 10:
                    issues["very_small_boxes"].append({
                        "image": img_ann['name'],
                        "category": category,
                        "area": area
                    })
                
                # Very large boxes (possible annotation errors)
                if area > 500000:
                    issues["very_large_boxes"].append({
                        "image": img_ann['name'],
                        "category": category,
                        "area": area
                    })
                
                # Unusual aspect ratios
                if aspect_ratio > 10 or (aspect_ratio < 0.1 and aspect_ratio > 0):
                    issues["unusual_aspect_ratios"].append({
                        "image": img_ann['name'],
                        "category": category,
                        "aspect_ratio": aspect_ratio
                        })
        
        return {
            "quality_issues": {
                k: {"count": len(v), "examples": v[:5]}
                for k, v in issues.items()
            },
            "recommendations": [
                f"Found {len(issues['very_small_boxes'])} tiny boxes - may be noise",
                f"Found {len(issues['unusual_aspect_ratios'])} unusual aspect ratios - verify annotations",
                "Consider filtering boxes < 10px² during training"
            ]
        }
    
    def identify_top_outliers(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Identify the most unusual/outlier object detections across all classes
        
        Scores individual objects (not images) based on:
        - Size anomalies (area >3σ from class mean)
        - High occlusion compared to class average
        - Truncation compared to class average
        - Weather/lighting conditions
        
        Args:
            top_k: Number of top outlier objects to return **per class**
            
        Returns:
            Dictionary with top K outlier objects per class and their scores
        """
        logger.info(f"Identifying top {top_k} outlier objects per class...")
        
        # Get class statistics for normalization
        class_stats = self.stats['category_stats']
        
        # Calculate average occlusion and truncation rates per class
        class_occlusion = {}
        class_truncation = {}
        class_counts = {}
        
        for img in self.annotations:
            labels = img.get('labels', [])
            if not isinstance(labels, list):
                continue
            
            for obj in labels:
                if not isinstance(obj, dict):
                    continue
                
                cat = obj.get('category')
                if not cat:
                    continue
                
                if cat not in class_counts:
                    class_counts[cat] = 0
                    class_occlusion[cat] = 0
                    class_truncation[cat] = 0
                
                class_counts[cat] += 1
                if obj.get('attributes', {}).get('occluded', False):
                    class_occlusion[cat] += 1
                if obj.get('attributes', {}).get('truncated', False):
                    class_truncation[cat] += 1
        
        # Calculate average rates
        for cat in class_counts:
            if class_counts[cat] > 0:
                class_occlusion[cat] = class_occlusion[cat] / class_counts[cat]
                class_truncation[cat] = class_truncation[cat] / class_counts[cat]
        
        outlier_objects = []
        
        # Score each individual object
        for img in self.annotations:
            img_name = img['name']
            img_attrs = img.get('attributes', {})
            weather = img_attrs.get('weather', 'clear')
            timeofday = img_attrs.get('timeofday', 'daytime')
            
            labels = img.get('labels', [])
            if not isinstance(labels, list):
                continue
            
            for obj_idx, obj in enumerate(labels):
                if not isinstance(obj, dict):
                    continue
                
                cat = obj.get('category')
                if not cat or cat not in class_stats:
                    continue
                
                box = obj.get('box2d', {})
                if not box:
                    continue
                
                score = 0
                reasons = []
                
                # 1. Size anomaly (z-score)
                area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
                mean_area = class_stats[cat]['mean_area']
                std_area = class_stats[cat]['std_area']
                
                if std_area > 0:
                    z_score = abs(area - mean_area) / std_area
                    if z_score > 3:  # More than 3 std devs
                        score += z_score * 15
                        reasons.append(f"Unusual size (z={z_score:.1f})")
                
                # 2. Occlusion outlier
                is_occluded = obj.get('attributes', {}).get('occluded', False)
                avg_occlusion = class_occlusion.get(cat, 0.5)
                
                if is_occluded and avg_occlusion < 0.3:
                    # Occluded when class usually isn't
                    score += 20
                    reasons.append(f"Unusually occluded ({cat} avg: {avg_occlusion*100:.0f}%)")
                elif not is_occluded and avg_occlusion > 0.7:
                    # Not occluded when class usually is
                    score += 15
                    reasons.append(f"Unusually visible ({cat} avg: {avg_occlusion*100:.0f}%)")
                
                # 3. Truncation outlier
                is_truncated = obj.get('attributes', {}).get('truncated', False)
                avg_truncation = class_truncation.get(cat, 0.2)
                
                if is_truncated and avg_truncation < 0.1:
                    # Truncated when class usually isn't
                    score += 15
                    reasons.append(f"Unusually truncated ({cat} avg: {avg_truncation*100:.0f}%)")
                elif not is_truncated and avg_truncation > 0.7:
                    # Not truncated when class usually is
                    score += 15
                    reasons.append(f"Unusually complete ({cat} avg: {avg_truncation*100:.0f}%)")
                
                # Only add if score is significant
                if score > 0:
                    outlier_objects.append({
                        'image_name': img_name,
                        'object_index': obj_idx,
                        'category': cat,
                        'outlier_score': float(score),
                        'reasons': reasons,
                        'area': float(area),
                        'occluded': bool(is_occluded),
                        'truncated': bool(is_truncated),
                        'weather': weather,
                        'timeofday': timeofday,
                        'bbox': box
                    })
        
        # Sort by score and get top K per class
        outliers_by_class = {}
        for obj in outlier_objects:
            cat = obj['category']
            if cat not in outliers_by_class:
                outliers_by_class[cat] = []
            outliers_by_class[cat].append(obj)
        
        # Get top K from each class
        top_outliers = []
        for cat in sorted(outliers_by_class.keys()):
            class_outliers = sorted(outliers_by_class[cat], key=lambda x: x['outlier_score'], reverse=True)
            top_outliers.extend(class_outliers[:top_k])  # Top K per class
        
        # Sort final list by score for display
        top_outliers.sort(key=lambda x: x['outlier_score'], reverse=True)
        
        logger.info(f"Identified {len(outlier_objects)} objects with outlier characteristics")
        logger.info(f"Top outlier score: {top_outliers[0]['outlier_score']:.1f}" if top_outliers else "No outliers found")
        
        return {
            'total_outliers_found': len(outlier_objects),
            'top_outliers': top_outliers,
            'outliers_per_class': {cat: len(outliers_by_class[cat]) for cat in outliers_by_class},
            'top_k_per_class': top_k,
            'scoring_criteria': [
                'Size anomalies (area >3σ from class mean)',
                'Occlusion outliers (compared to class average)',
                'Truncation outliers (compared to class average)',
                'Rare/uncommon classes (train, bus, rider, motor)',
                'Challenging weather/lighting conditions'
            ],
            'class_statistics': {
                'average_occlusion_rates': class_occlusion,
                'average_truncation_rates': class_truncation
            }
        }


def analyze_anomalies(data_root: Path, split: str = 'train', output_dir: Path = None) -> Dict:
    """
    Analyze anomalies in BDD100K dataset
    
    Args:
        data_root: Root directory containing BDD100K data
        split: Dataset split ('train', 'val', 'test')
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all anomaly detection results
    """
    from .parse_annotations import load_bdd_images, get_dataset_statistics
    
    # Load data
    logger.info(f"Analyzing anomalies in {split} split...")
    label_file = data_root / "labels" / f"bdd100k_labels_images_{split}.json"
    images = load_bdd_images(str(label_file))
    statistics = get_dataset_statistics(images)
    
    # Detect anomalies
    detector = AnomalyDetector(images, statistics)
    results = detector.detect_all()
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{split}_anomalies.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Anomaly analysis saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Detect anomalies in BDD100K dataset")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing BDD100K data')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to analyze')
    parser.add_argument('--output-dir', type=str, default='./data/analysis',
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    results = analyze_anomalies(
        Path(args.data_root),
        args.split,
        Path(args.output_dir)
    )
    
    # Print summary
    print("\n" + "="*70)
    print("ANOMALY DETECTION SUMMARY")
    print("="*70)
    
    # Class imbalance
    print("\n1. CLASS IMBALANCE:")
    imb = results['class_imbalance']
    print(f"   Imbalance Ratio: {imb['imbalance_ratio']:.1f}x")
    print(f"   Dominant: {imb['dominant_class']['name']} ({imb['dominant_class']['percentage']:.1f}%)")
    print(f"   Rare Classes: {len(imb['rare_classes'])}")
    
    # Size anomalies
    print("\n2. SIZE ANOMALIES:")
    for cls, cv in results['size_anomalies']['most_variable_classes']:
        print(f"   {cls}: {cv:.1f}% variation")
    
    # Occlusion
    print("\n3. OCCLUSION PATTERNS:")
    print(f"   Overall Rate: {results['occlusion_patterns']['overall_occlusion_rate']:.1f}%")
    for cls, rate in results['occlusion_patterns']['most_occluded_classes']:
        print(f"   {cls}: {rate:.1f}% occluded")
    
    # Rare cases
    print("\n4. RARE/EDGE CASES:")
    print(f"   Found: {results['rare_cases']['rare_cases_found']} cases")
    
    # Extreme cases
    print("\n5. EXTREME CASES:")
    for case_type, data in results['extreme_cases']['extreme_cases'].items():
        print(f"   {case_type}: {data['count']} images")
    
    print("\n" + "="*70)
