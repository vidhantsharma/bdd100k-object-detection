# BDD100K Object Detection Pipeline

End-to-end object detection pipeline for BDD100K dataset with Faster R-CNN fine-tuning.


---

## 🚀 Quick Start

### 1. Run Docker Container

```bash
# Set your BDD100K dataset path (or edit run_docker.sh to set BDD100K_DATA_PATH)
export BDD100K_DATA_PATH=/path/to/your/bdd100k

# Start Docker (auto-builds on first run, rebuilds with --rebuild flag)
./run_docker.sh
```

### 2. Inside Docker: Create Subset

```bash
# Create balanced subset (1000 train + 200 val images)
# Recommended for faster training and experimentation
python3 -m src.utils.create_subset --train-size 1000 --val-size 200
```

**Note:** This step is required for training. The subset ensures:
- Balanced class representation (all 10 classes present)
- Faster training iterations (~2-3 hours vs 20+ hours on full dataset)
- Quality-filtered images (minimum 2 objects per image)

To use the full dataset (~70K images), modify `config/config.yaml`:
```yaml
data:
  root: "data/raw"  # Instead of "data/subset"
```

### 3. Run Pipeline Stages

```bash
# Analyze dataset
python3 main.py --stage analysis --split train

# Train model (10 epochs, ~2-3 hours on GPU)
python3 main.py --stage train

# Evaluate fine-tuned model (if available)
python3 main.py --stage evaluate

# Evaluate zero-shot (pretrained COCO model, no fine-tuning)
python3 main.py --stage evaluate --zero-shot
```

---

## 📊 Analysis Results

Comprehensive dataset analysis on 1000 training images:

### Class Distribution
| Class | Count | % | Challenge |
|-------|-------|---|-----------|
| car | 10,113 | 50.0% | Most common |
| traffic sign | 3,777 | 18.7% | Small size |
| traffic light | 3,061 | 15.1% | Small size |
| person | 1,866 | 9.2% | Occlusion |
| truck | 483 | 2.4% | - |
| bus | 254 | 1.3% | - |
| bike | 244 | 1.2% | High occlusion |
| motor | 143 | 0.7% | High occlusion |
| rider | 207 | 1.0% | High occlusion |
| train | 76 | 0.4% | Very rare |

### Key Statistics
- **Total objects**: 20,224
- **Avg objects/image**: 20.2
- **Occlusion rate**: 47.3% (high)
- **Class imbalance**: 133x ratio (car vs train)

### View Analysis Dashboards

After running analysis, open the interactive dashboards in your browser:

```bash
# On your host machine (outside Docker)
firefox data/analysis/train_dashboard.html
# or
google-chrome data/analysis/train_dashboard.html
# or
xdg-open data/analysis/train_dashboard.html
```

> **Note**: If the HTML files don't open, ensure they have read permissions:
> ```bash
> chmod -R 644 data/analysis/*.html
> chmod -R 755 data/analysis/
> ```

**Dashboards include:**
- `train_dashboard.html` - Interactive visualizations (class distribution, bbox sizes, occlusion rates)
- `training_insights.html` - Training recommendations and anomalies
- `visualizations/` - Sample images per class, outlier visulization

**Outputs**: `data/analysis/` - Statistics, dashboards, visualizations

---

## 🎓 Model: Faster R-CNN + ResNet50-FPN

### Why This Model?

**Faster R-CNN with ResNet50-FPN** was chosen as the optimal architecture for BDD100K autonomous driving dataset:

#### 1. **Two-Stage Detection = Higher Accuracy**
Unlike one-stage detectors (YOLO, SSD, RetinaNet), Faster R-CNN separates object localization and classification:
- **Stage 1 (RPN)**: Region Proposal Network generates hundreds to a few thousand candidate proposals (config-dependent, typically ~2000)
- **Stage 2**: RoI Align + Classification head refines boxes and predicts classes
- **Result**: More accurate localization, especially for overlapping objects (common in BDD100K - 47% occlusion rate)

#### 2. **Feature Pyramid Network (FPN) = Multi-Scale Detection**
BDD100K has **extreme scale variation** (133x size difference):
- **Tiny objects**: Traffic lights (~200px²), distant traffic signs
- **Large objects**: Trucks (>100,000px²), buses
- **FPN solution**: Builds feature pyramids with lateral connections, detecting objects at multiple resolutions
- **Why better**: Simpler backbones without explicit multi-scale fusion (e.g., plain ResNet/VGG) struggle with small objects, while FPN provides explicit multi-resolution supervision

#### 3. **ResNet50 Backbone = Proven Performance**
- **41M parameters**: Good balance between speed and capacity
- **Residual connections**: Enables training deeper networks without vanishing gradients
- **ImageNet pretraining**: Learns general features (edges, textures, shapes)
- **Faster than**: ResNet101 (2x slower), better than ResNet18 (less capacity)

#### 4. **Transfer Learning Strategy**
- **Start**: COCO pretrained weights (80 classes including car, person, truck, bus, traffic light)
- **Fine-tune**: Replace classification head with 10 BDD100K classes, train on subset
- **Why not train from scratch?** Requires 10x more data and training time for similar performance
- **Domain adaptation**: COCO → BDD100K provides a strong initialization, but still requires fine-tuning due to domain shift (dashcam viewpoint, weather conditions, scale differences)

#### 5. **BDD100K-Specific Advantages**
- ✅ **Handles occlusion**: Two-stage refinement better than one-stage anchor regression
- ✅ **Class imbalance robustness**: Can use weighted sampling (implemented)
- ✅ **Dashcam viewpoint**: COCO cars/persons help with domain transfer
- ✅ **Near-real-time**: ~5–10 FPS on single GPU (RTX 3050), suitable for offline perception, validation pipelines, or low-speed AV stacks

#### Why Not Other Models?
| Model | Why Not? |
|-------|----------|
| **YOLO** | Single-stage, poor on small objects (traffic lights/signs) |
| **SSD** | Fixed aspect ratios, struggles with extreme scales |
| **RetinaNet** | Better than YOLO, but still one-stage limitations |
| **Mask R-CNN** | Overkill (no need for segmentation masks), slower |
| **DETR** | Transformer-based, requires 10x more training data |

### Architecture
```
Input Image (720×1280)
    ↓
ResNet50 Backbone (ImageNet pretrained)
    ↓
Feature Pyramid Network (multi-scale feature maps P2–P5)
    ↓
Region Proposal Network (hundreds to a few thousand candidate regions)
    ↓
RoI Align (7×7 pooled features, preserves spatial alignment)
    ↓
Box Regression Head + Classification Head
    ↓
Non-Maximum Suppression (proposal and final detection stages)
    ↓
Output: bounding boxes, class labels, confidence scores
```

**Model Size**: ~41M parameters | **Throughput**: ~5–10 FPS on RTX 3050 (batch size 1)

---

## 📈 Evaluation Results

Evaluated on 201 validation images from subset.

### Zero-Shot (Pretrained COCO, No Fine-Tuning)

| Class | AP | Notes |
|-------|-----|-------|
| car | 7.2% | Best (COCO has cars) |
| person | 3.6% | Viewpoint mismatch |
| traffic light | 5.2% | Small objects |
| truck | 1.4% | Poor |
| rider | 0.0% | No COCO equivalent |
| traffic sign | 0.0% | No COCO equivalent |
| bike | 0.0% | Poor |
| motor | 0.0% | Poor |
| bus | 0.0% | Poor |
| train | 0.0% | Poor |
| **mAP@0.5** | **1.74%** | Baseline |

**Why so bad?** Domain mismatch (COCO→BDD100K), no traffic sign class in COCO, dashcam viewpoint.

### Fine-Tuned (10 Epochs on Subset)

| Class | AP | Improvement |
|-------|-----|-------------|
| car | 63.1% | +55.9% ✅ |
| traffic light | 50.0% | +44.8% ✅ |
| motor | 42.7% | +42.7% ✅ |
| traffic sign | 41.3% | +41.3% ✅ |
| rider | 39.1% | +39.1% ✅ |
| bus | 38.5% | +38.5% ✅ |
| truck | 38.0% | +36.6% ✅ |
| bike | 33.8% | +33.8% ✅ |
| train | 3.5% | +3.5% ✅ |
| person | 0.0% | -3.6% ❌ |
| **mAP@0.5** | **34.99%** | **+20x improvement** 🚀 |

**Key findings:**
- ✅ **Massive improvement** after fine-tuning (1.74% → 34.99%)
- ✅ **Car detection excellent** (63.1% AP) - most common class
- ✅ **Good recall** (60-78%) - model finds objects
- ⚠️ **Low precision** - many false positives
- ❌ **Person class failed** - training issue (class imbalance or catastrophic forgetting)

**Outputs**: `data/evaluation/results.json`, `data/evaluation/visualizations/`

---

## 📁 Project Structure

```
bdd100k-object-detection/
├── run_docker.sh              # Main entry point - starts Docker container
├── docker/
│   ├── Dockerfile             # Docker image definition
│   ├── docker-compose.yml     # Container configuration (GPU, volumes)
│   ├── entrypoint.sh          # Auto-setup: creates data/raw structure from mounted data
│   └── .dockerignore          # Excludes data/ from build context
├── main.py                    # Pipeline orchestrator (analysis, train, evaluate)
├── config/
│   └── config.yaml            # All hyperparameters and settings
├── src/
│   ├── analysis/              # Dataset analysis (stats, anomalies, dashboards)
│   ├── datasets/              # BDD100K PyTorch Dataset and transforms
│   ├── dataloaders/           # DataLoader with weighted sampling
│   ├── models/                # Faster R-CNN model wrapper
│   ├── training/              # Training loop, loss, optimizer
│   ├── evaluation/            # mAP metrics, visualization
│   └── utils/
│       └── create_subset.py   # Creates balanced subset from full dataset
├── data/                      # Created automatically, saved to host
│   ├── raw/                   # Symbolic links to mounted BDD100K data
│   ├── subset/                # Selected 1000 train + 200 val images
│   ├── analysis/              # Statistics, dashboards, visualizations
│   ├── training/              # Checkpoints: best_model.pth, latest_checkpoint.pth
│   └── evaluation/            # results.json, prediction visualizations
└── requirements.txt           # Python dependencies
```

### Key Files

| File | Purpose |
|------|---------|
| `run_docker.sh` | Builds Docker image, starts container with GPU and volumes |
| `docker/entrypoint.sh` | Auto-detects nested BDD100K structure, creates flat data/raw/ |
| `main.py` | Runs analysis, training, or evaluation stages |
| `src/utils/create_subset.py` | Creates balanced subset ensuring all classes present |
| `src/training/train.py` | Training loop with checkpointing, class weighting |
| `src/evaluation/metrics.py` | Computes mAP@0.5, per-class AP |
| `src/models/detector.py` | Faster R-CNN wrapper with BDD100K class mapping |
| `config/config.yaml` | Training config (epochs: 10, batch_size: 2, lr: 0.005) |

---

## 🔧 Configuration

Edit `config/config.yaml` to customize training:

```yaml
model:
  pretrained: true              # Fine-tune (true) or train from scratch (false)
  num_classes: 10
  backbone:
    trainable_layers: 3         # Fine-tune last 3 ResNet blocks

training:
  batch_size: 2                 # GPU memory: RTX 3050 4GB
  num_epochs: 10
  learning_rate: 0.005
  use_class_weights: true       # Handle class imbalance
  
evaluation:
  iou_threshold: 0.5            # mAP@0.5
  confidence_threshold: 0.0
```

---

## 🐛 Troubleshooting

**"Docker permission denied"**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**"CUDA out of memory"**
```yaml
# In config.yaml
training:
  batch_size: 1  # Reduce from 2
```

**"Rebuild after changing Docker files"**
```bash
./run_docker.sh --rebuild
```

---

## 📚 References

- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [BDD100K Paper](https://arxiv.org/abs/1805.04687)
