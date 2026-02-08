# BDD100K Object Detection - Complete Pipeline

Comprehensive end-to-end pipeline for BDD100K object detection including data analysis, preprocessing, model training, and evaluation.

## 🎯 Features

### 📊 Dataset Analysis
- Class distribution and imbalance analysis
- Object count distribution per image
- Bounding box size statistics (mean, std, quartiles)
- Occlusion and truncation rates
- Weather and time-of-day distribution
- Anomaly detection (rare classes, size anomalies, quality issues)
- Interactive HTML dashboards with Plotly visualizations
- Training insights and recommendations

### 🔧 Data Preprocessing
- BDD100K to COCO format conversion
- Data validation and filtering
- Minimum bbox area thresholding
- Image dimension verification

### 🎓 Model Training (PyTorch)
- **Model**: Faster R-CNN with ResNet50-FPN backbone
- **Fine-tuning**: Pretrained on COCO dataset
- **Train from scratch**: Optional (set `model.pretrained: false` in config)
- **Class imbalance handling**:
  - Weighted loss based on class frequencies
  - Focal loss support
- **Features**:
  - Automatic checkpointing (best + periodic saves)
  - TensorBoard logging
  - Learning rate scheduling (step, multistep, cosine)
  - Gradient clipping
  - Resume from checkpoint

### 📈 Evaluation
- Mean Average Precision (mAP) @ IoU=0.5
- Per-class Average Precision
- Visualization of predictions
- Test on validation/test sets or single images

## 🏗️ Architecture: Faster R-CNN + ResNet50-FPN

### Why This Model?

**Faster R-CNN** is the optimal choice for BDD100K because:

1. **Two-Stage Detection** → Better accuracy than one-stage detectors
   - Stage 1: Region Proposal Network (RPN) proposes candidate boxes
   - Stage 2: Classification and bbox refinement
   - mAP on BDD100K: ~0.45 vs YOLO's ~0.35

2. **Feature Pyramid Network (FPN)** → Handles extreme scale variation
   - BDD100K has huge scale differences:
     - Traffic lights: tiny (~200px²)
     - Trucks/buses: large (>100,000px²)
   - FPN detects objects at multiple scales efficiently

3. **ResNet50 Backbone** → Good balance of speed vs accuracy
   - Pretrained on ImageNet (general features)
   - Can be fine-tuned (default) or trained from scratch
   - 41M parameters

4. **Proven Performance** → Research shows best results on BDD100K
   - Autonomous driving prioritizes accuracy over speed
   - Robust to weather/lighting variations
   - Handles occlusion and truncation well

### Model Configuration

```yaml
model:
  architecture: "faster_rcnn_resnet50_fpn"
  pretrained: true              # Fine-tune (true) or train from scratch (false)
  num_classes: 10               # BDD100K object classes
  backbone:
    trainable_layers: 3         # Number of ResNet layers to fine-tune (0-5)
```

## 📁 Project Structure

```
bdd100k-object-detection/
├── main.py                     # Main pipeline (analysis, training, evaluation)
├── train.py                    # Standalone training script
├── test.py                     # Standalone testing script
├── quickstart.py               # Setup verification
├── config/
│   └── config.yaml            # Configuration file
├── data/
│   ├── raw/                   # Original BDD100K data
│   │   ├── images/            # train/val/test subdirs
│   │   └── labels/            # JSON annotation files
│   └── analysis/              # Generated analysis outputs
├── outputs/
│   ├── training/              # Training checkpoints & logs
│   └── evaluation/            # Evaluation results
└── src/
    ├── analysis/              # Data analysis modules
    ├── preprocessing/         # Data conversion
    ├── datasets/              # PyTorch Dataset
    ├── dataloaders/           # DataLoader builders
    ├── models/                # Model definitions
    ├── training/              # Training pipeline
    ├── evaluation/            # Metrics & evaluation
    └── utils/                 # Utilities
```
## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Data

Download BDD100K dataset and extract to:
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

### 3. Verify Setup

```bash
python3 quickstart.py
```

### 4. Run Pipeline Stages

#### **Stage 1: Data Analysis**
```bash
# Analyze training data
python3 main.py --stage analysis --split train

# Analyze validation data
python3 main.py --stage analysis --split val
```

**Outputs** (in `data/analysis/`):
- `train_statistics.json` - Comprehensive statistics
- `train_anomalies.json` - Detected anomalies
- `train_dashboard.html` - Interactive visualizations
- `training_insights.html` - Training recommendations
- `visualizations/train_top_outliers.png` - Outlier images

#### **Stage 2: Data Preprocessing**
```bash
# Preprocess training data (converts to COCO format)
python3 main.py --stage preprocessing --split train
```

**Outputs**: COCO format JSON in `data/processed/`

#### **Stage 3: Model Training**
```bash
# Train using main.py (recommended - uses config)
python3 main.py --stage train

# Or use standalone training script
python3 train.py --data-root data/raw --num-epochs 20 --batch-size 4
```

**Note**: `--split` is NOT used for training. Training automatically uses both train and val splits:
- Train split: for model training
- Val split: for validation during training

**Training Options:**
- **Fine-tune** (default): `model.pretrained: true` in config
  - Faster convergence (~10-20 epochs)
  - Better performance with limited data
  - ✅ Recommended for BDD100K
  
- **Train from scratch**: `model.pretrained: false` in config
  - Requires more epochs (~50+)
  - Needs more computational resources
  - Use if you want complete control

**Class Imbalance Handling:**
- **Weighted loss** (default): `training.use_class_weights: true`
  - Automatically computed from data analysis
  - Weights inversely proportional to class frequency
  
- **Focal loss**: `training.loss_type: "focal"`
  - Focuses on hard examples
  - Good for extreme imbalance

**Outputs** (in `outputs/training/`):
- `best_model.pth` - Best checkpoint (by validation loss)
- `final_model.pth` - Final epoch model
- `checkpoint_epoch_*.pth` - Periodic saves
- `training_history.json` - Loss curves
- `logs/` - TensorBoard logs

#### **Stage 4: Model Evaluation**
```bash
# Evaluate using main.py
python3 main.py --stage evaluate

# Or use standalone test script
python3 test.py \
    --model-path outputs/training/best_model.pth \
    --data-root data/raw \
    --split val \
    --evaluate \
    --visualize
```

**Note**: Evaluation uses validation split by default.

**Outputs** (in `outputs/evaluation/`):
- `evaluation_results.json` - mAP and metrics
- `visualizations/` - Prediction images

#### **Run Complete Pipeline**
```bash
python3 main.py --stage all
```

Runs: Analysis (train) → Preprocessing (train) → Training → Evaluation (val)

### 5. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir outputs/training/logs

# Open browser at: http://localhost:6006
```

### 6. Test on Single Image

```bash
python3 test.py \
    --model-path outputs/training/best_model.pth \
    --image-path data/raw/images/val/your_image.jpg \
    --output-path prediction.png \
    --confidence-threshold 0.5
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Model Settings
```yaml
model:
  pretrained: true              # Fine-tune (true) or train from scratch (false)
  num_classes: 10
  backbone:
    trainable_layers: 3         # Fine-tune last 3 ResNet blocks
```

### Training Settings
```yaml
training:
  batch_size: 4
  num_epochs: 20
  learning_rate: 0.005
  optimizer: "sgd"
  
  # Class imbalance handling
  use_class_weights: true       # Use weights from data analysis
  loss_type: "weighted"         # Options: "weighted", "focal", "standard"
  
  # Learning rate scheduling
  scheduler:
    type: "step"                # Decay every 3 epochs
    step_size: 3
    gamma: 0.1
```

### Evaluation Settings
```yaml
evaluation:
  iou_threshold: 0.5
  confidence_threshold: 0.5
```

## 📊 BDD100K Dataset

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

## 📊 BDD100K Dataset

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
- **car**: 55.4% (most common) - 713K instances
- **traffic sign**: 18.6% - 240K instances
- **traffic light**: 14.5% - 186K instances
- **person**: 7.1% - 91K instances
- **truck**: 2.3% - 30K instances
- **bus**: 0.9% - 12K instances
- **bike**: 0.6% - 7K instances
- **rider**: 0.4% - 5K instances
- **motor**: 0.2% - 3K instances
- **train**: 0.01% (rarest) - 136 instances

**Challenge**: 5,244x ratio between most and least common classes!

## 🔬 Handling Class Imbalance

Your data analysis identified severe class imbalance. We handle this with:

### 1. Weighted Loss (Default)
```yaml
training:
  use_class_weights: true
  loss_type: "weighted"
```

**How it works:**
- Automatically computes weights from data analysis
- Weight = `total_objects / (num_classes * class_count)`
- Rare classes get higher weights (e.g., train: 94.6x, car: 0.18x)
- Applied to classification loss component

**When to use:**
- Default choice for BDD100K
- Balances class importance
- Fast and effective

### 2. Focal Loss
```yaml
training:
  loss_type: "focal"
```

**How it works:**
- Focuses on hard-to-classify examples
- Down-weights easy examples
- Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`

**When to use:**
- Extreme imbalance (like train class)
- When standard weighted loss isn't enough
- Slightly slower than weighted loss

### 3. Standard Loss
```yaml
training:
  use_class_weights: false
  loss_type: "standard"
```

**When to use:**
- Balanced datasets
- Comparison baseline
- Not recommended for BDD100K

## 📈 Expected Performance

| Configuration | Epochs | Time (RTX 3050) | mAP@0.5 | Notes |
|--------------|--------|-----------------|---------|-------|
| Fine-tune (pretrained) | 10 | ~2-3 hours | 0.25-0.35 | Baseline |
| Fine-tune (pretrained) | 20 | ~4-6 hours | 0.35-0.45 | Recommended |
| Fine-tune (pretrained) | 50+ | ~10-15 hours | 0.45-0.55 | Best |
| Train from scratch | 50+ | ~12-20 hours | 0.30-0.40 | Not recommended |

**Per-Class Performance (typical):**
- **Good**: car (0.65), person (0.55), traffic sign (0.50)
- **Moderate**: bus (0.45), truck (0.40), traffic light (0.38)
- **Challenging**: rider (0.30), bike (0.28), motor (0.20)
- **Difficult**: train (0.10-0.15) - very rare class

## 🎓 Model Architecture Details

### Faster R-CNN Pipeline

```
Input Image (720×1280)
     ↓
ResNet50 Backbone (pretrained on ImageNet)
     ↓ (Extract features at multiple levels)
Feature Pyramid Network (FPN)
     ↓ (Multi-scale features: P2, P3, P4, P5, P6)
Region Proposal Network (RPN)
     ↓ (Generate ~1000-2000 proposals)
RoI Align (Extract features for each proposal)
     ↓
Fast R-CNN Head
     ↓
[Classification + Bbox Regression]
     ↓
Output: boxes, labels, scores
```

### Components:

1. **ResNet50 Backbone**
   - 50 layers deep
   - Pretrained on ImageNet
   - Extracts feature maps at multiple scales
   - Can freeze early layers, fine-tune later layers

2. **Feature Pyramid Network (FPN)**
   - Builds multi-scale feature pyramid
   - P2 (stride 4): small objects (traffic lights)
   - P3-P5: medium objects (cars, persons)
   - P6: large objects (trucks, buses)

3. **Region Proposal Network (RPN)**
   - Proposes candidate object locations
   - Uses anchors at multiple scales and aspect ratios
   - Reduces proposals from millions to thousands

4. **Fast R-CNN Head**
   - Classification: predicts object class
   - Bbox regression: refines bbox coordinates
   - Applies class weights for imbalance

### Why Not YOLO or SSD?

| Model | mAP on BDD100K | Speed | Why Not? |
|-------|----------------|-------|----------|
| YOLOv5 | ~0.35 | Fast | Lower accuracy on small objects |
| SSD | ~0.30 | Fast | Struggles with scale variation |
| Faster R-CNN | ~0.45 | Moderate | ✓ Best accuracy, handles scales |

**For autonomous driving**: Accuracy > Speed

## 💡 Key Insights from Analysis
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
