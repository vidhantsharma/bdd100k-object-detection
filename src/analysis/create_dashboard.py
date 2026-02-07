"""
Interactive Dashboard for BDD100K Dataset Visualization

This creates an HTML dashboard with interactive plots showing:
- Class distribution
- Size distributions
- Occlusion patterns
- Weather and time conditions
- Spatial distributions
- Interesting samples
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List
import numpy as np
from string import Template


def get_class_order(stats: Dict) -> List[str]:
    """Get consistent class ordering (by count, descending) for all charts"""
    category_counts = stats['category_counts']
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    return [c[0] for c in sorted_cats]

def create_class_distribution_chart(stats: Dict) -> go.Figure:
    """Create interactive bar chart of class distribution"""
    category_counts = stats['category_counts']
    total = sum(category_counts.values())
    
    # Sort by count
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    categories = [c[0] for c in sorted_cats]
    counts = [c[1] for c in sorted_cats]
    percentages = [100 * c / total for c in counts]
    
    # Format percentages with appropriate precision
    text_labels = []
    for p in percentages:
        if p >= 1.0:
            text_labels.append(f'{p:.1f}%')
        elif p >= 0.1:
            text_labels.append(f'{p:.2f}%')
        else:
            text_labels.append(f'{p:.3f}%')
    
    fig = go.Figure([go.Bar(
        x=categories,
        y=counts,
        text=text_labels,
        textposition='outside',
        marker=dict(
            color=counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Count")
        ),
        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{text}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Class Distribution (Log Scale)",
        xaxis_title="Class",
        yaxis_title="Number of Objects",
        yaxis_type="log",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_size_distribution_chart(stats: Dict) -> go.Figure:
    """Create box plot of object size distributions"""
    fig = go.Figure()
    
    # Use same class order as class distribution chart
    class_order = get_class_order(stats)
    
    for category in class_order:
        if category not in stats['category_stats']:
            continue
        cat_stats = stats['category_stats'][category]
        min_area = cat_stats['min_area']
        median = cat_stats.get('median_area', cat_stats['mean_area'])
        max_area = cat_stats['max_area']
        
        # Create a simple box plot with min, median, max
        fig.add_trace(go.Box(
            y=[min_area, median, max_area],
            name=category,
            boxmean=False,
            hovertemplate=(
                f'<b>{category}</b><br>'
                'Min: %{customdata[0]:,.0f} px²<br>'
                'Median: %{customdata[1]:,.0f} px²<br>'
                'Max: %{customdata[2]:,.0f} px²<br>'
                '<extra></extra>'
            ),
            customdata=[[min_area, median, max_area]] * 3
        ))
    
    fig.update_layout(
        title="Object Size Distribution by Class",
        xaxis_title="Class",
        yaxis_title="Area (px²)",
        yaxis_type="log",
        height=500,
        showlegend=False
    )
    
    return fig


def create_occlusion_chart(annotations: List[Dict], stats: Dict) -> go.Figure:
    """Create stacked bar chart of occlusion rates by class (object detection classes only)"""
    from collections import defaultdict
    
    # Classes to exclude (segmentation/non-object-detection classes)
    EXCLUDE_CLASSES = {'drivable area', 'lane'}
    
    occlusion_data = defaultdict(lambda: {"occluded": 0, "visible": 0})
    
    for img in annotations:
        for obj in img.get('labels', []):
            cat = obj.get('category')
            if cat and cat not in EXCLUDE_CLASSES:
                if obj.get('attributes', {}).get('occluded', False):
                    occlusion_data[cat]["occluded"] += 1
                else:
                    occlusion_data[cat]["visible"] += 1
    
    # Use same class order as class distribution chart
    class_order = get_class_order(stats)
    categories = [c for c in class_order if c in occlusion_data]
    occluded = [occlusion_data[c]["occluded"] for c in categories]
    visible = [occlusion_data[c]["visible"] for c in categories]
    
    fig = go.Figure(data=[
        go.Bar(name='Visible', x=categories, y=visible, marker_color='lightblue'),
        go.Bar(name='Occluded', x=categories, y=occluded, marker_color='coral')
    ])
    
    fig.update_layout(
        title="Occlusion Patterns by Class (Log Scale)",
        xaxis_title="Class",
        yaxis_title="Number of Objects",
        yaxis_type="log",  # Add log scale
        barmode='stack',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_weather_timeofday_chart(stats: Dict) -> go.Figure:
    """Create subplots for weather and time of day distributions"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Weather Distribution", "Time of Day Distribution"),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )
    
    # Weather pie chart
    weather_data = stats['weather_distribution']
    fig.add_trace(
        go.Pie(labels=list(weather_data.keys()), values=list(weather_data.values()),
               name="Weather", hole=0.3),
        row=1, col=1
    )
    
    # Time of day pie chart
    timeofday_data = stats['timeofday_distribution']
    fig.add_trace(
        go.Pie(labels=list(timeofday_data.keys()), values=list(timeofday_data.values()),
               name="Time", hole=0.3),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig


def create_scene_distribution_chart(stats: Dict) -> go.Figure:
    """Create horizontal bar chart for scene types"""
    scene_data = stats['scene_distribution']
    
    # Sort by count
    sorted_scenes = sorted(scene_data.items(), key=lambda x: -x[1])
    scenes = [s[0] for s in sorted_scenes]
    counts = [s[1] for s in sorted_scenes]
    
    fig = go.Figure([go.Bar(
        y=scenes,
        x=counts,
        orientation='h',
        marker_color='teal',
        text=[f'{c:,}' for c in counts],
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Scene Type Distribution",
        xaxis_title="Number of Images",
        yaxis_title="Scene Type",
        height=400
    )
    
    return fig


def create_objects_per_image_histogram(stats: Dict, annotations: list = None) -> go.Figure:
    """Create histogram of objects per image
    
    Shows the distribution of how many objects appear in each image across the dataset.
    This helps understand image complexity and annotation density.
    """
    if annotations:
        object_counts = []
        for img in annotations:
            count = len([obj for obj in img.get('labels', []) 
                        if obj.get('category') not in {'drivable area', 'lane'}])
            object_counts.append(count)
        
        fig = go.Figure([go.Histogram(
            x=object_counts,
            nbinsx=50,
            marker_color='purple',
            name='Images',
            hovertemplate='<b>%{x} objects</b><br>%{y} images<br><extra></extra>'
        )])
    else:
        # Fallback: show summary stats as bar chart
        fig = go.Figure([go.Bar(
            x=['Min', 'Median', 'Mean', 'Max'],
            y=[stats['min_objects_per_image'], 
               stats['median_objects_per_image'],
               stats['avg_objects_per_image'], 
               stats['max_objects_per_image']],
            marker_color='purple',
            name='Statistics'
        )])
    
    # Add vertical lines for key statistics (if we have the distribution)
    if annotations:
        mean_val = stats['avg_objects_per_image']
        median_val = stats['median_objects_per_image']
        
        fig.add_vline(x=mean_val, line_dash="dash",
                      annotation_text=f"Mean: {mean_val:.1f}",
                      annotation_position="top right",
                      line_color="red", line_width=2)
        fig.add_vline(x=median_val, line_dash="dash",
                      annotation_text=f"Median: {median_val}",
                      annotation_position="top left",
                      line_color="green", line_width=2)
    
    fig.update_layout(
        title="Objects per Image Distribution<br><sub>Shows how many detection objects appear in each image (excludes segmentation classes)</sub>",
        xaxis_title="Number of Objects per Image",
        yaxis_title="Number of Images" if annotations else "Count",
        height=400,
        showlegend=False
    )
    
    return fig


def create_dashboard(data_root: Path, split: str = 'train', output_file: str = None):
    """
    Create interactive HTML dashboard
    
    Args:
        data_root: Root directory containing BDD100K data
        split: Dataset split ('train', 'val')
        output_file: Output HTML file path
    """
    print(f"Creating dashboard for {split} split...")
    
    # Load data
    stats_file = Path('data/analysis') / f'{split}_statistics.json'
    with open(stats_file) as f:
        stats = json.load(f)
    
    label_file = data_root / 'labels' / f'bdd100k_labels_images_{split}.json'
    with open(label_file) as f:
        annotations = json.load(f)
    
    print("Generating visualizations...")
    
    # Create all charts
    class_dist = create_class_distribution_chart(stats)
    size_dist = create_size_distribution_chart(stats)
    occlusion_chart = create_occlusion_chart(annotations, stats)
    weather_time_chart = create_weather_timeofday_chart(stats)
    scene_chart = create_scene_distribution_chart(stats)
    objects_hist = create_objects_per_image_histogram(stats, annotations)
    
    # Load HTML template
    template_path = Path(__file__).parent / 'templates' / 'dashboard_template.html'
    with open(template_path) as f:
        template = Template(f.read())
    
    # Determine output path first
    if output_file is None:
        output_file = f'data/analysis/{split}_dashboard.html'
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if outlier visualization exists
    outlier_viz_path = output_path.parent / 'visualizations' / f'{split}_top_outliers.png'
    outlier_link = f"visualizations/{split}_top_outliers.png" if outlier_viz_path.exists() else "#"
    
    # Prepare template variables
    template_vars = {
        'split_upper': split.upper(),
        'split_lower': split.lower(),
        'total_images': f"{stats['total_images']:,}",
        'total_objects': f"{stats['total_objects']:,}",
        'avg_objects': f"{stats['avg_objects_per_image']:.1f}",
        'occlusion_rate': f"{stats['occlusion_rate']*100:.1f}%",
        'num_classes': len(stats['category_counts']),
        'outlier_viz_path': outlier_link,
        
        # Chart data and layouts as JSON strings
        'class_dist_data': json.dumps(class_dist.to_dict()),
        'class_dist_layout': json.dumps(class_dist.to_dict()),
        'size_dist_data': json.dumps(size_dist.to_dict()),
        'size_dist_layout': json.dumps(size_dist.to_dict()),
        'occlusion_data': json.dumps(occlusion_chart.to_dict()),
        'occlusion_layout': json.dumps(occlusion_chart.to_dict()),
        'weather_time_data': json.dumps(weather_time_chart.to_dict()),
        'weather_time_layout': json.dumps(weather_time_chart.to_dict()),
        'scene_data': json.dumps(scene_chart.to_dict()),
        'scene_layout': json.dumps(scene_chart.to_dict()),
        'objects_hist_data': json.dumps(objects_hist.to_dict()),
        'objects_hist_layout': json.dumps(objects_hist.to_dict()),
    }
    
    # Render template
    html_content = template.safe_substitute(template_vars)
    
    # Save dashboard (output_path already defined above)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Dashboard created: {output_path}")
    print(f"   Open it in your browser to explore the data!")
    
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create BDD100K analysis dashboard")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing BDD100K data')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path')
    
    args = parser.parse_args()
    
    dashboard_path = create_dashboard(
        Path(args.data_root),
        args.split,
        args.output
    )
    
    print(f"\n🌐 Open the dashboard: file://{Path(dashboard_path).absolute()}")
