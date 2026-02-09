# Model Evaluation and Performance Analysis

**Date**: February 9, 2026  
**Model**: Faster R-CNN ResNet50-FPN (COCO pretrained, zero-shot)  
**Dataset**: BDD100K validation set (100 images)  
**Evaluation Metric**: mAP@IoU=0.5 (COCO-style 101-point interpolation)

---

## 1. Quantitative Performance Summary

### Overall Performance
```
mAP@0.5: 2.41%
```

**Interpretation**: Extremely poor performance, indicating severe domain mismatch between COCO training data and BDD100K driving scenes.

### Per-Class Results

| Class          | AP (%) | Precision | Recall | F1    | Predictions | Ground Truth |
|----------------|--------|-----------|--------|-------|-------------|--------------|
| **car**        | 8.50   | 0.245     | 0.122  | 0.163 | 497         | 999          |
| **traffic light** | 7.70 | 0.223   | 0.120  | 0.156 | 179         | 334          |
| **person**     | 4.74   | 0.290     | 0.071  | 0.114 | 69          | 281          |
| **truck**      | 3.19   | 0.364     | 0.054  | 0.094 | 11          | 74           |
| **rider**      | 0.00   | 0.000     | 0.000  | 0.000 | 0           | 32           |
| **bus**        | 0.00   | 0.000     | 0.000  | 0.000 | 1           | 27           |
| **bike**       | 0.00   | 0.000     | 0.000  | 0.000 | 0           | 46           |
| **motor**      | 0.00   | 0.000     | 0.000  | 0.000 | 0           | 14           |
| **traffic sign** | 0.00 | 0.000   | 0.000  | 0.000 | 0           | 384          |
| **train**      | 0.00   | 0.000     | 0.000  | 0.000 | 1           | 11           |

---

## 2. Analysis: What Works and What Doesn't

### ✅ What Works (Relatively)

#### 1. **Car Detection** (AP: 8.5%)
- **Why it works**: 
  - Cars are abundant in COCO dataset
  - Direct class mapping (COCO car → BDD car)
  - Large, distinctive objects
- **Limitations**:
  - Still poor (8.5% vs >40% for fine-tuned models)
  - Struggles with distant/occluded cars
  - 497 predictions but 999 GT → severe under-detection

#### 2. **Traffic Light Detection** (AP: 7.7%)
- **Why it partially works**:
  - Mapped from COCO "stop sign" class (imperfect match)
  - Distinctive red/green colors
- **Limitations**:
  - Traffic lights are tiny in dashcam footage (~200px²)
  - Different perspective than COCO stop signs
  - 179 predictions vs 334 GT → misses 46% of traffic lights

#### 3. **Person Detection** (AP: 4.7%)
- **Why it has some success**:
  - Common in COCO dataset
  - Highest precision (0.29) among all classes
- **Limitations**:
  - Only detects 20/281 people (7% recall!)
  - Misses pedestrians at crosswalks (small, distant)

### ❌ What Doesn't Work

#### 1. **Traffic Signs** (AP: 0%, 0 predictions!)
- **Root cause**: **No COCO equivalent class**
  - No class mapping exists
  - Model never learned to detect traffic signs
- **Impact**: Most critical failure
  - 384 GT traffic signs in validation set (19% of objects)
  - Complete blind spot for autonomous driving

#### 2. **Bikes/Motorcycles** (AP: 0%)
- **Root causes**:
  - Poor COCO→BDD mapping ("bicycle" vs "bike", "motorcycle" vs "motor")
  - Small, often occluded objects
  - Uncommon viewing angles (side view from dashcam)
- **Data from Stage 1 analysis**:
  - Bikes: 2.3% of dataset (class imbalance)
  - Motors: 0.7% of dataset (very rare)

#### 3. **Buses** (AP: 0%, only 1 prediction)
- **Root causes**:
  - Infrequent in COCO dataset
  - Severe occlusion and truncation in BDD100K
  - Similar appearance to trucks (confuses model)
- **Evidence**: Only 1 prediction for 27 GT buses

---

## 3. Failure Pattern Analysis

### Failure Cluster 1: **Small Objects** (Traffic lights, signs, distant vehicles)
**Characteristics**:
- Object area < 1000px²
- Confidence scores typically < 0.1
- Recall < 15% across all small object classes

**Why it fails**:
- COCO objects are typically close-up (20-40% of image)
- BDD100K objects are distant (1-5% of image)
- Model's anchor boxes not optimized for tiny objects

**Evidence from visualizations**:
- Traffic light examples show GT boxes with NO red prediction boxes
- Most small objects completely missed

### Failure Cluster 2: **Class Mapping Gaps** (Traffic signs, bikes, motors)
**Characteristics**:
- No direct COCO equivalent
- 0 predictions despite hundreds of GT boxes
- Complete detection failure

**Why it fails**:
- COCO has 91 classes but missing key driving classes
- No "traffic sign" class in COCO
- "Bicycle" in COCO ≠ "bike" in BDD (different contexts)

**Evidence**:
- Traffic signs: 0/384 detected
- Bikes: 0/46 detected
- Motors: 0/14 detected

### Failure Cluster 3: **Domain Shift** (Weather, lighting, motion blur)
**Characteristics**:
- Low confidence scores (most < 0.3)
- High false negative rate
- Precision ~25% (3 in 4 predictions are wrong)

**Why it fails**:
- **COCO**: Indoor/outdoor scenes, static images, good lighting
- **BDD100K**: Dashcam footage, motion blur, rain/fog, night driving

**Evidence from Stage 1 analysis**:
- 23% of images in rainy/snowy/foggy weather
- 14% night/dawn/dusk lighting
- Model trained on clear, daytime images

### Failure Cluster 4: **Severe Occlusion & Truncation**
**Characteristics**:
- Recall drops dramatically for occluded objects
- Truncated objects at image edges missed

**Evidence from Stage 1 anomaly detection**:
- 31% of objects have occlusion
- 29% of objects are truncated
- Model likely trained on complete, unoccluded COCO objects

---

## 4. Metric Justification

### Why mAP@0.5?
1. **Industry standard** for object detection
2. **COCO evaluation protocol** (comparing to baseline)
3. **Single number** summarizing performance across classes
4. **IoU=0.5** balances precision (too strict if 0.75) and leniency (too loose if 0.3)

### Why Per-Class Metrics?
1. **Identify weak classes** (traffic signs, bikes, buses)
2. **Guide training priorities** (focus on failing classes)
3. **Class imbalance insights** (connect to data analysis)

### Why Precision, Recall, F1?
1. **Precision**: How many predictions are correct? (low = many false positives)
2. **Recall**: How many GT objects detected? (low = missing objects)
3. **F1**: Harmonic mean balancing both (low = poor overall)

**Our results show**:
- Low precision (0.24) → Many false positives (detecting objects that aren't there)
- Very low recall (0.12) → Severe under-detection (missing most real objects)
- F1 = 0.16 → Extremely poor balance

---

## 5. Connection to Data Analysis (Stage 1)

### Finding 1: **Severe Class Imbalance**
**From Stage 1**:
- Cars: 51% of objects
- Traffic signs: 19% of objects
- Bikes: 2.3% of objects

**Impact on Performance**:
- Car AP (8.5%) > Bike AP (0%)
- Model benefits from COCO's car abundance
- Rare classes (bikes, buses) suffer without weighted loss

**Recommendation**: Use **focal loss** or **class weights** during training

---

### Finding 2: **Small Object Dominance**
**From Stage 1**:
- Mean bbox area: 8,456 px²
- Traffic lights: ~200 px² (tiny!)
- 25% of objects < 2,000 px²

**Impact on Performance**:
- Traffic light AP (7.7%) < Car AP (8.5%)
- Recall for small objects < 12%
- Confidence scores for small objects < 0.1

**Recommendation**: 
- Use **Feature Pyramid Networks** (already in model but needs tuning)
- **Multi-scale training augmentation**
- **Anchor box redesign** for tiny objects

---

### Finding 3: **High Occlusion & Truncation**
**From Stage 1 anomaly detection**:
- 31% occlusion rate
- 29% truncation rate
- These are "hard examples"

**Impact on Performance**:
- Low recall (0.12) → Missing occluded/truncated objects
- Model trained on clean COCO images struggles with real-world clutter

**Recommendation**:
- **Data augmentation**: Random erasing, cutout, mixup
- **Train on hard examples**: Oversample occluded/truncated objects

---

### Finding 4: **Weather & Lighting Variations**
**From Stage 1**:
- 23% rainy/foggy/snowy images
- 14% night/dawn/dusk images
- 63% clear/daytime images

**Impact on Performance**:
- Low confidence scores (mean ~0.25)
- Domain shift from COCO's clear images

**Recommendation**:
- **Color augmentation**: Brightness, contrast, saturation
- **Weather simulation**: Add fog/rain/snow effects
- **Train on diverse lighting conditions**

---

## 6. Visualization Analysis

### Qualitative Observations from Per-Class Visualizations

#### **Sample Analysis: Traffic Light Examples**
- **Ground Truth** (Green boxes): 3-5 tiny traffic lights per image
- **Predictions** (Red boxes): 0-1 detections, mostly missed
- **Failure mode**: Model blind to small, distant traffic lights
- **Root cause**: Anchor boxes too large, feature resolution too coarse

#### **Sample Analysis: Car Examples**
- **Ground Truth** (Green boxes): 10-20 cars per image
- **Predictions** (Red boxes): 3-5 detections (under-detection)
- **Failure mode**: Misses distant cars, focuses on closest vehicles
- **Pattern**: Better at large, foreground cars than small, background cars

#### **Sample Analysis: Person Examples**
- **Ground Truth** (Green boxes): 2-5 pedestrians per image
- **Predictions** (Red boxes): 0-1 detections
- **Failure mode**: Misses pedestrians at crosswalks, sidewalks
- **Root cause**: Small size, motion blur, occlusion

---

## 7. Suggested Improvements

### Priority 1: **Fine-Tune on BDD100K** (Expected: 35-45% mAP)
- Train for 10 epochs on BDD100K train set
- Use pretrained COCO weights as initialization
- Expected improvement: **2.4% → 40% mAP** (17x improvement)

### Priority 2: **Address Class Imbalance**
```yaml
training:
  use_class_weights: true
  loss_type: "focal"  # Instead of weighted
```
- **Focal loss**: Focuses on hard examples, down-weights easy negatives
- **Expected impact**: +5-10% mAP, especially for rare classes

### Priority 3: **Multi-Scale Training**
```yaml
training:
  augmentation:
    random_resize:
      min_size: 400
      max_size: 1200  # Increase from 1000
    multi_scale: [800, 1000, 1200]
```
- Trains model to handle objects at various scales
- **Expected impact**: +3-5% mAP for small objects

### Priority 4: **Better Data Augmentation**
```yaml
training:
  augmentation:
    color_jitter: true
    random_fog: true  # Simulate weather
    random_shadow: true
    mixup: true  # Mix two images
```
- Reduces domain shift
- **Expected impact**: +2-3% mAP, better generalization

### Priority 5: **Anchor Box Optimization**
- Current anchors: [32, 64, 128, 256, 512]
- BDD100K needs smaller anchors: [16, 32, 64, 128, 256]
- **Expected impact**: +5-8% mAP for traffic lights/signs

---

## 8. Conclusion

### Summary of Findings
1. **Zero-shot COCO→BDD transfer fails** (2.4% mAP vs >40% fine-tuned)
2. **Domain mismatch** is the primary bottleneck (dashcam vs human perspective)
3. **Small objects** (traffic lights, signs) are the biggest weakness
4. **Class mapping gaps** (no traffic sign class) cause complete failures
5. **Fine-tuning is essential** for any practical performance

### Next Steps
1. ✅ Complete Stage 1 (Data Analysis)
2. ✅ Complete Stage 2 (Preprocessing)
3. ⏳ **Run Stage 3** (Training) → Expected: ~40% mAP
4. ⏳ **Re-evaluate** with fine-tuned model
5. ⏳ **Iterate** on augmentation, loss functions, architecture

### Expected Performance After Training
```
Current (zero-shot):  mAP = 2.4%
After fine-tuning:    mAP = 35-45% (realistic goal)
SOTA on BDD100K:      mAP = 45-50% (with heavy optimization)
```

---

## Appendix: Visualization Samples

**Generated Files**:
- `data/evaluation/visualizations/sample_predictions.png` - 6 diverse examples
- `data/evaluation/visualizations/per_class/` - 9 per-class visualizations
  - `car_examples.png` (best performing)
  - `traffic_light_examples.png` (small object challenges)
  - `traffic_sign_examples.png` (complete failure, 0 detections)
  - `person_examples.png` (severe under-detection)
  - And 5 more classes...

**Color Coding**:
- 🟢 **Green solid lines** = Ground Truth
- 🔴 **Red dashed lines** = Model Predictions
- **Confidence threshold**: 0.05 (low to show all predictions)

**Key Observations**:
- Most images have 5-20 green GT boxes but only 0-3 red prediction boxes
- Traffic sign examples show NO red boxes (complete detection failure)
- Car examples show some red boxes but miss most cars (severe under-detection)
