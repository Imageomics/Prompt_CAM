# Reidac9 Dataset Visualization Guide

This guide explains how to create visualizations for the Reidac9 dataset using Prompt_CAM.

## Dataset Information

The Reidac9 dataset contains 9 classes:
- **0**: Exterior
- **1**: Interior  
- **2**: another
- **3**: bathroom
- **4**: bedroom
- **5**: dining_room
- **6**: document
- **7**: kitchen
- **8**: living_room

## Prerequisites

1. Trained model checkpoint (`.pth` file)
2. Reidac9 dataset in the correct location
3. Python environment with required dependencies

## Quick Start

### Method 1: Using the Python Script (Recommended)

```bash
cd Prompt_CAM

# Visualize all classes
python visualize_reidac9.py --checkpoint <path_to_checkpoint>

# Visualize specific classes (e.g., bedroom, kitchen, living_room)
python visualize_reidac9.py --checkpoint <path_to_checkpoint> --classes "4,7,8"

# Customize number of samples and traits
python visualize_reidac9.py \
    --checkpoint <path_to_checkpoint> \
    --classes "all" \
    --nmbr_samples 20 \
    --top_traits 6
```

### Method 2: Using the Shell Script

```bash
cd Prompt_CAM
./visualize_reidac9.sh <path_to_checkpoint>
```

### Method 3: Manual Visualization (Single Class)

```bash
cd Prompt_CAM

# Visualize a specific class
python visualize.py \
    --config vis_config_reidac9.yaml \
    --checkpoint <path_to_checkpoint> \
    --vis_cls 4 \
    --nmbr_samples 10 \
    --top_traits 4
```

## Configuration

The visualization configuration is stored in `vis_config_reidac9.yaml`. You can modify:

- `vis_cls`: Class to visualize (0-8)
- `nmbr_samples`: Number of samples per class (default: 10)
- `top_traits`: Number of top attention heads to show (default: 4)
- `vis_outdir`: Output directory (default: ./visualization)

## Output Structure

Visualizations will be saved in:
```
./visualization/dino/reidac9/
├── class_0/
│   └── top_traits_4/
│       ├── img_1/
│       ├── img_2/
│       └── ...
├── class_1/
├── class_2/
└── ...
```

Each image folder contains:
- Original image
- Attention maps for top traits
- Combined visualization

## Examples

### Visualize Bedroom Images
```bash
python visualize_reidac9.py --checkpoint model.pth --classes "4" --nmbr_samples 15
```

### Visualize Kitchen and Living Room
```bash
python visualize_reidac9.py --checkpoint model.pth --classes "7,8" --nmbr_samples 20 --top_traits 6
```

### Visualize All Interior Spaces
```bash
python visualize_reidac9.py --checkpoint model.pth --classes "3,4,5,7,8"
```

## Troubleshooting

### Checkpoint Not Found
Make sure your checkpoint path is correct:
```bash
# Check if file exists
ls -lh <path_to_checkpoint>
```

### CUDA Out of Memory
Reduce batch size or number of samples:
```bash
python visualize_reidac9.py --checkpoint model.pth --nmbr_samples 5
```

### Dataset Path Issues
Ensure the dataset is in the correct location. The script expects:
- `reidac9_data/reidac9_list_train.txt`
- `reidac9_data/reidac9_list_test.txt`
- `reidac9_data/Compress/` (image directory)

## Advanced Usage

### Custom Configuration
Create your own config file based on `vis_config_reidac9.yaml`:

```yaml
data: reidac9
data_path: ./
vis_cls: 4
nmbr_samples: 20
top_traits: 6
vis_outdir: ./my_visualizations
```

Then run:
```bash
python visualize.py --config my_custom_config.yaml --checkpoint model.pth
```

### Batch Processing
To process multiple checkpoints:

```bash
for ckpt in checkpoints/*.pth; do
    echo "Processing $ckpt"
    python visualize_reidac9.py --checkpoint "$ckpt" --classes "4,7,8"
done
```

## Notes

- Visualization requires a trained model checkpoint
- The first run may take longer as it loads the model
- GPU is recommended for faster processing
- Results are saved automatically in the output directory
