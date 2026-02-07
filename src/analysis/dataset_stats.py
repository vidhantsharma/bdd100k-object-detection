"""Dataset statistics computation and reporting."""
import json
from pathlib import Path
from typing import Any, Dict


def save_statistics_report(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Save statistics report to JSON file.
    
    Args:
        stats: Statistics dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Statistics saved to: {output_path}")


def print_statistics_summary(stats: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of statistics.
    
    Args:
        stats: Statistics dictionary
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 60)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Images: {stats['total_images']:,}")
    print(f"  Total Objects: {stats['total_objects']:,}")
    print(f"  Avg Objects per Image: {stats['avg_objects_per_image']:.2f}")
    print(f"  Median Objects per Image: {stats['median_objects_per_image']:.2f}")
    print(f"  Max Objects per Image: {stats['max_objects_per_image']}")
    print(f"  Min Objects per Image: {stats['min_objects_per_image']}")
    
    print(f"\nOcclusion & Truncation:")
    print(f"  Occlusion Rate: {stats['occlusion_rate']:.2%}")
    print(f"  Truncation Rate: {stats['truncation_rate']:.2%}")
    
    print(f"\nClass Distribution:")
    print(f"  {'Class':<20} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}")
    
    sorted_classes = sorted(
        stats['category_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for cls, count in sorted_classes:
        percentage = count / stats['total_objects'] * 100
        print(f"  {cls:<20} {count:>8,} {percentage:>11.2f}%")
    
    if stats.get('weather_distribution'):
        print(f"\nWeather Distribution:")
        for weather, count in sorted(stats['weather_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['total_images'] * 100
            print(f"  {weather:<20} {count:>8,} {percentage:>11.2f}%")
    
    if stats.get('timeofday_distribution'):
        print(f"\nTime of Day Distribution:")
        for time, count in sorted(stats['timeofday_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['total_images'] * 100
            print(f"  {time:<20} {count:>8,} {percentage:>11.2f}%")
    
    print("\n" + "=" * 60)


