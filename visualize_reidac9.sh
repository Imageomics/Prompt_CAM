#!/bin/bash

# Visualization script for Reidac9 dataset
# This script generates visualizations for all 9 classes

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: ./visualize_reidac9.sh <path_to_checkpoint>"
    echo "Example: ./visualize_reidac9.sh ./checkpoints/reidac9_model.pth"
    exit 1
fi

CHECKPOINT=$1

# Class names for reference
# 0: Exterior
# 1: Interior
# 2: another
# 3: bathroom
# 4: bedroom
# 5: dining_room
# 6: document
# 7: kitchen
# 8: living_room

echo "Starting visualization for Reidac9 dataset..."
echo "Checkpoint: $CHECKPOINT"
echo ""

# Visualize each class
for class_id in {0..8}; do
    echo "Visualizing class $class_id..."
    python visualize.py \
        --config vis_config_reidac9.yaml \
        --checkpoint $CHECKPOINT \
        --vis_cls $class_id \
        --nmbr_samples 10 \
        --top_traits 4
    echo "Class $class_id completed."
    echo ""
done

echo "All visualizations completed!"
echo "Results saved in: ./visualization/dino/reidac9/"
