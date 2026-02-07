# BDD100K Object Detection Dataset Analysis

Comprehensive analysis pipeline for the BDD100K object detection dataset. Generates statistics, identifies anomalies, detects outliers, and creates interactive visualizations and training insights.

## Features

### 📊 Dataset Statistics
- Class distribution and imbalance analysis
- Object count distribution per image
- Bounding box size statistics
- Occlusion and truncation rates
- Weather and time-of-day distribution

### 🔍 Anomaly Detection
- **Class Imbalance Analysis**: Identifies rare classes (< 1% of dataset)
- **Size Anomalies**: Detects classes with high size variation (coefficient of variation)
- **Occlusion Patterns**: Finds most occluded classes and truncation rates
- **Quality Issues**: Identifies tiny boxes (<10px²), large boxes (>500k px²), unusual aspect ratios

### 🎯 Outlier Detection
- Per-class outlier identification
- Scoring based on:
  - Size anomalies (deviation from class mean)
  - Occlusion patterns
  - Truncation patterns
- Visual grid showing top K outliers per class

### 📈 Interactive Dashboard
- 6 interactive Plotly visualizations
- Class distribution bar chart
- Object count per image histogram
- Average bbox size by class
- Occlusion/truncation rates
- Weather distribution pie chart
- Time-of-day distribution pie chart
- Clickable links to outlier visualization and training insights

### 💡 Training Insights Page
- Actionable recommendations for model training
- 4 comprehensive analysis sections:
  1. **Class Imbalance Analysis**: Rare classes and rebalancing strategies
  2. **Size Anomalies & Variation**: Multi-scale detection recommendations
  3. **Occlusion & Truncation Patterns**: Handling partial visibility
  4. **Annotation Quality Issues**: Filtering noisy labels

## Project Structure

```
bdd100k-object-detection-analysis/
├── main.py                   # Main analysis pipeline
├── config/
│   └── config.yaml          # Configuration (paths, thresholds)
├── data/
│   ├── raw/                 # Original BDD100K data
│   │   ├── images/          # train/val/test subdirs
│   │   └── labels/          # JSON annotation files
│   └── analysis/            # Generated outputs
│       ├── train_statistics.json
│       ├── train_anomalies.json
│       ├── train_dashboard.html
│       ├── training_insights.html
│       └── visualizations/
│           └── train_top_outliers.png
└── src/
    └── analysis/            # Analysis modules
        ├── parse_annotations.py
        ├── dataset_stats.py
        ├── detect_anomalies.py
        ├── visualize_samples.py
        ├── create_dashboard.py
        └── create_insights.py
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bdd100k-object-detection-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download BDD100K Dataset

1. Register at [BDD100K website](https://bdd-data.berkeley.edu/)
2. Download the object detection dataset:
   - Images: `bdd100k_images_100k.zip`
   - Labels: `bdd100k_labels_release.zip`
3. Extract to `data/raw/`:

```bash
# Extract images
unzip bdd100k_images_100k.zip -d data/raw/
# Extract labels
unzip bdd100k_labels_release.zip -d data/raw/
```

Expected structure:
```
data/raw/
├── images/
│   ├── train/      (70K images)
│   ├── val/        (10K images)
│   └── test/       (20K images)
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

## Usage

### Run Complete Analysis Pipeline

```bash
python main.py --data-root data/raw --split train --stages analysis --top-k 4
```

**Arguments:**
- `--data-root`: Root directory containing BDD100K data
- `--split`: Dataset split to analyze (`train`, `val`, `test`)
- `--stages`: Pipeline stages to run (`analysis`, `conversion`, `training`, `evaluation`)
- `--top-k`: Number of top outliers per class (default: 3)

### Output Files

After running, the following files are generated in `data/analysis/`:

1. **`train_statistics.json`** - Dataset statistics and distributions
2. **`train_anomalies.json`** - Detected anomalies and outliers
3. **`train_dashboard.html`** - Interactive dashboard with 6 charts
4. **`training_insights.html`** - Training recommendations page
5. **`visualizations/train_top_outliers.png`** - Visual grid of outlier images

### View Results

Open the dashboard in your browser:

```bash
# On Linux/Mac
xdg-open data/analysis/train_dashboard.html

# On Windows
start data/analysis/train_dashboard.html

# Or manually open in browser
firefox data/analysis/train_dashboard.html
```

## BDD100K Dataset

### Object Classes (10 categories)
1. **person** - Pedestrians
2. **rider** - Person on bike/motorcycle
3. **car** - Passenger vehicles
4. **bus** - Buses
5. **truck** - Trucks and large vehicles
6. **bike** - Bicycles
7. **motor** - Motorcycles
8. **traffic light** - Traffic signals
9. **traffic sign** - Road signs
10. **train** - Trains (rarest class: 0.01%)

### Dataset Statistics (Train Split)
- **Total Images**: 69,863
- **Total Objects**: 1,286,871
- **Avg Objects/Image**: 18.42
- **Occlusion Rate**: 47.3%
- **Truncation Rate**: 6.9%

### Class Distribution
- **car**: 55.4% (most common)
- **traffic sign**: 18.6%
- **traffic light**: 14.5%
- **person**: 7.1%
- **truck**: 2.3%
- **bus**: 0.9%
- **bike**: 0.6%
- **rider**: 0.4%
- **motor**: 0.2%
- **train**: 0.01% (rarest)

## Key Insights

### 1. Severe Class Imbalance
- 5,244x ratio between most and least common classes
- Rare classes (train, motor, rider) need special handling
- **Recommendations**: Class-balanced sampling, focal loss, oversampling

### 2. High Size Variation
- Traffic signs/lights have 350+ coefficient of variation
- Objects appear at multiple scales (near vs. far)
- **Recommendations**: Feature Pyramid Networks (FPN), multi-scale anchors

### 3. High Occlusion Rates
- 47.3% of objects are occluded
- Rider (89.2%), bike (83.8%), motor (76.5%) most occluded
- **Recommendations**: Part-based detection, attention mechanisms, occlusion augmentation

### 4. Annotation Quality Issues
- Small boxes (<10px²): Potential noise
- Large boxes (>500k px²): May be scene-level labels
- Unusual aspect ratios: Possible annotation errors
- **Recommendations**: Filter or review suspicious annotations

## Configuration

The project uses a centralized configuration file: `config/config.yaml`

### Key Settings

```yaml
# Global settings
global:
  seed: 42
  logging:
    level: INFO

# Data paths
data:
  root: data/raw
  output_dir: data/analysis

# Analysis parameters
analysis:
  anomaly_detection:
    rare_class_threshold: 0.01    # Classes < 1% of dataset
    tiny_box_threshold: 10        # Boxes < 10px²
    large_box_threshold: 500000   # Boxes > 500k px²
  
  outlier_detection:
    top_k: 3                      # Outliers per class
    scoring:
      size_deviation_weight: 15   # Points per std deviation
      occlusion_weight: 20        # Occlusion anomaly points
      truncation_weight: 15       # Truncation anomaly points
```

### Usage

**Use default config:**
```bash
python main.py --split train --stages analysis
```

**Override config values:**
```bash
python main.py --split train --stages analysis --top-k 5 --data-root custom/path
```

**Use custom config file:**
```bash
python main.py --config my_config.yaml --split train --stages analysis
```

### Extensibility

The config file is designed to be extended for future features:
- **Model Configuration**: Architecture, backbone, hyperparameters
- **Training Configuration**: Batch size, learning rate, optimizer settings
- **Evaluation Configuration**: Metrics, thresholds, visualization options

This allows you to add training/evaluation stages without modifying code!

## Development

The codebase is modular and well-documented:

- **`parse_annotations.py`**: Load and parse BDD100K JSON format
- **`dataset_stats.py`**: Compute statistics and distributions
- **`detect_anomalies.py`**: Anomaly and outlier detection algorithms
- **`visualize_samples.py`**: Create outlier visualizations
- **`create_dashboard.py`**: Generate interactive Plotly dashboard
- **`create_insights.py`**: Generate training insights page

All functions are called by `main.py` - no unused code!

## References

- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [BDD100K Paper](https://arxiv.org/abs/1805.04687)
- [CVPR 2020 Publication](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html)

## Citation

If you use BDD100K dataset, please cite:

```bibtex
@inproceedings{bdd100k,
  author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and 
            Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

## License

This analysis tool is provided for educational and research purposes. 
Please refer to the BDD100K dataset license for data usage terms.
