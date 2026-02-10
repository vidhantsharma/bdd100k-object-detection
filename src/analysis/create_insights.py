#!/usr/bin/env python3
"""
Generate training insights page from dataset analysis
"""

from typing import Dict, Any
from pathlib import Path
from string import Template
import json


def generate_insights_html(anomalies: Dict[str, Any], output_path: str) -> None:
    """
    Generate HTML page with training insights and recommendations
    
    Args:
        anomalies: Dictionary with analysis results from detect_all()
        output_path: Path to save the HTML file
    """
    
    # Load HTML template
    template_path = Path(__file__).parent / 'templates' / 'insights_template.html'
    with open(template_path) as f:
        template = Template(f.read())
    
    # Generate sections
    class_imbalance = generate_class_imbalance_section(anomalies.get('class_imbalance', {}))
    size_anomalies = generate_size_anomalies_section(anomalies.get('size_anomalies', {}))
    occlusion_patterns = generate_occlusion_patterns_section(anomalies.get('occlusion_patterns', {}))
    quality_issues = generate_quality_issues_section(anomalies.get('quality_issues', {}))
    
    # Render template
    html_content = template.safe_substitute(
        class_imbalance_section=class_imbalance,
        size_anomalies_section=size_anomalies,
        occlusion_patterns_section=occlusion_patterns,
        quality_issues_section=quality_issues
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Training insights page saved to: {output_path}")


def generate_class_imbalance_section(data: Dict[str, Any]) -> str:
    """Generate HTML for class imbalance insights"""
    if not data:
        return ""
    
    imbalance_ratio = data.get('imbalance_ratio', 0)
    rare_classes = data.get('rare_classes', [])
    recommendations = data.get('recommendations', [])
    
    rare_classes_html = ''.join([f'<span class="class-tag">{cls}</span>' for cls in rare_classes])
    recommendations_html = ''.join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])
    
    return f"""
        <div class="section">
            <h2><span class="section-icon">⚖️</span>Class Imbalance Analysis</h2>
            
            <div class="key-finding">
                The dataset shows a <strong>{imbalance_ratio:.0f}x imbalance</strong> between the most and least common classes.
                This severe imbalance can lead to poor detection of rare classes.
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Imbalance Ratio</div>
                    <div class="stat-value">{imbalance_ratio:.0f}x</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Rare Classes Found</div>
                    <div class="stat-value">{len(rare_classes)}</div>
                </div>
            </div>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Rare Classes (< 1% of dataset)</h3>
            <div class="class-list">
                {rare_classes_html}
            </div>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Training Recommendations</h3>
            {recommendations_html}
        </div>
    """


def generate_size_anomalies_section(data: Dict[str, Any]) -> str:
    """Generate HTML for size anomalies insights"""
    if not data:
        return ""
    
    variable_classes = data.get('most_variable_classes', [])
    recommendations = data.get('recommendations', [])
    
    if not variable_classes:
        return ""
    
    # Create table rows for variable classes
    table_rows = ""
    for cls_data in variable_classes[:10]:  # Top 10
        # Handle [class_name, cv_value] format
        if isinstance(cls_data, (list, tuple)) and len(cls_data) >= 2:
            cls_name = cls_data[0]
            cv = cls_data[1]
        else:
            continue
            
        table_rows += f"""
            <tr>
                <td><strong>{cls_name}</strong></td>
                <td><span class="metric-value">{cv:.2f}</span></td>
                <td>High size variation indicates objects at different distances/scales</td>
            </tr>
        """
    
    recommendations_html = ''.join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])
    
    return f"""
        <div class="section">
            <h2><span class="section-icon">📏</span>Size Anomalies & Variation</h2>
            
            <div class="key-finding">
                Identified <strong>{len(variable_classes)} classes</strong> with high size variation.
                Objects appearing at multiple scales require multi-scale detection strategies.
            </div>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Classes with High Size Variation</h3>
            <table class="insights-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Coefficient of Variation</th>
                        <th>Implication</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Training Recommendations</h3>
            {recommendations_html}
        </div>
    """


def generate_occlusion_patterns_section(data: Dict[str, Any]) -> str:
    """Generate HTML for occlusion patterns insights"""
    if not data:
        return ""
    
    # Use the correct key names from detect_anomalies.py
    occlusion_rate = data.get('overall_occlusion_rate', 0)
    most_occluded = data.get('most_occluded_classes', [])
    truncation_rates = data.get('truncation_rates', {})
    recommendations = data.get('recommendations', [])
    
    # Calculate overall truncation rate from per-class rates
    if truncation_rates:
        truncation_rate = sum(truncation_rates.values()) / len(truncation_rates)
    else:
        truncation_rate = 0
    
    # Create table for most occluded classes
    table_rows = ""
    for cls_data in most_occluded[:10]:
        if (isinstance(cls_data, (list, tuple)) and len(cls_data) >= 2):
            cls_name = cls_data[0]
            rate = cls_data[1]
        else:
            continue
            
        table_rows += f"""
            <tr>
                <td><strong>{cls_name}</strong></td>
                <td><span class="metric-value">{rate:.1f}%</span></td>
                <td>Frequently partially hidden - requires robust occlusion handling</td>
            </tr>
        """
    
    recommendations_html = ''.join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])
    
    return f"""
        <div class="section">
            <h2><span class="section-icon">👁️</span>Occlusion & Truncation Patterns</h2>
            
            <div class="key-finding">
                <strong>{occlusion_rate:.1f}%</strong> of objects are occluded and <strong>{truncation_rate:.1f}%</strong> are truncated.
                Models must handle partial visibility effectively.
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Occlusion Rate</div>
                    <div class="stat-value">{occlusion_rate:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Truncation Rate</div>
                    <div class="stat-value">{truncation_rate:.1f}%</div>
                </div>
            </div>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Most Occluded Classes</h3>
            <table class="insights-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Occlusion Rate</th>
                        <th>Training Consideration</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Training Recommendations</h3>
            {recommendations_html}
        </div>
    """


def generate_quality_issues_section(data: Dict[str, Any]) -> str:
    """Generate HTML for annotation quality issues"""
    if not data:
        return ""
    
    quality_issues = data.get('quality_issues', {})
    recommendations = data.get('recommendations', [])
    
    if not quality_issues:
        return ""
    
    # Count totals
    small_boxes = quality_issues.get('very_small_boxes', {}).get('count', 0)
    large_boxes = quality_issues.get('very_large_boxes', {}).get('count', 0)
    unusual_ar = quality_issues.get('unusual_aspect_ratios', {}).get('count', 0)
    
    total_issues = small_boxes + large_boxes + unusual_ar
    
    recommendations_html = ''.join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])
    
    return f"""
        <div class="section">
            <h2><span class="section-icon">🔍</span>Annotation Quality Issues</h2>
            
            <div class="key-finding">
                Found <strong>{total_issues:,}</strong> potential annotation quality issues.
                Review and filter these before training to avoid learning from noisy labels.
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Very Small Boxes (&lt;10 px²)</div>
                    <div class="stat-value">{small_boxes:,}</div>
                    <div style="font-size: 11px; color: #666; margin-top: 8px;">Boxes smaller than 3×3 pixels - likely annotation errors or distant objects</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Very Large Boxes (&gt;500k px²)</div>
                    <div class="stat-value">{large_boxes:,}</div>
                    <div style="font-size: 11px; color: #666; margin-top: 8px;">Boxes covering ~45% of 1280×720 image - may be background/scene labels</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Unusual Aspect Ratios</div>
                    <div class="stat-value">{unusual_ar:,}</div>
                    <div style="font-size: 11px; color: #666; margin-top: 8px;">Width/height ratio &gt;10 (thin vertical) or &lt;0.1 (thin horizontal) - e.g., 1×50px or 50×1px boxes</div>
                </div>
            </div>
            
            <h3 style="margin-top: 25px; margin-bottom: 15px; color: #333; font-size: 18px;">Training Recommendations</h3>
            {recommendations_html}
        </div>
    """
