#!/bin/bash
echo "========================================"
echo "BDD100K Data Setup"
echo "========================================"

# Remove any existing raw data structure
rm -rf /workspace/data/raw

if [ -d "/data/mounted" ]; then
    echo "✓ Found mounted BDD100K data at /data/mounted"
    
    # Create data/raw directory structure
    mkdir -p /workspace/data/raw/images
    mkdir -p /workspace/data/raw/labels
    
    # Detect nested vs flat structure
    if [ -d "/data/mounted/bdd100k_images_100k/bdd100k/images/100k" ]; then
        echo "✓ Detected nested BDD100K structure"
        IMAGES_BASE="/data/mounted/bdd100k_images_100k/bdd100k/images/100k"
        LABELS_BASE="/data/mounted/bdd100k_labels_release/bdd100k/labels"
    elif [ -d "/data/mounted/images" ]; then
        echo "✓ Detected flat structure"
        IMAGES_BASE="/data/mounted/images"
        LABELS_BASE="/data/mounted/labels"
    else
        echo "❌ Error: Could not detect BDD100K data structure"
        exit 1
    fi
    
    # Create symbolic links for images
    [ -d "$IMAGES_BASE/train" ] && ln -sf "$IMAGES_BASE/train" /workspace/data/raw/images/train
    [ -d "$IMAGES_BASE/val" ] && ln -sf "$IMAGES_BASE/val" /workspace/data/raw/images/val
    [ -d "$IMAGES_BASE/test" ] && ln -sf "$IMAGES_BASE/test" /workspace/data/raw/images/test
    
    # Create symbolic links for labels
    [ -f "$LABELS_BASE/bdd100k_labels_images_train.json" ] && ln -sf "$LABELS_BASE/bdd100k_labels_images_train.json" /workspace/data/raw/labels/bdd100k_labels_images_train.json
    [ -f "$LABELS_BASE/bdd100k_labels_images_val.json" ] && ln -sf "$LABELS_BASE/bdd100k_labels_images_val.json" /workspace/data/raw/labels/bdd100k_labels_images_val.json
    
    echo "✓ Data structure created at /workspace/data/raw"
else
    echo "❌ Warning: No data mounted at /data/mounted"
fi

echo "========================================"
echo "Output directories (saved to host):"
echo "  /workspace/data/subset     -> ./data/subset/"
echo "  /workspace/data/analysis   -> ./data/analysis/"
echo "  /workspace/data/evaluation -> ./data/evaluation/"
echo "  /workspace/data/training   -> ./data/training/"
echo "========================================"
echo ""

exec "$@"
