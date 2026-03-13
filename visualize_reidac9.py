#!/usr/bin/env python3
"""
Visualization script for Reidac9 dataset
Generates attention visualizations for the 9 classes in the dataset
"""

import argparse
import os
import subprocess

# Class mapping for Reidac9
CLASS_NAMES = {
    0: 'Exterior',
    1: 'Interior',
    2: 'another',
    3: 'bathroom',
    4: 'bedroom',
    5: 'dining_room',
    6: 'document',
    7: 'kitchen',
    8: 'living_room'
}

def visualize_class(checkpoint, class_id, num_samples=10, top_traits=4, output_dir='./visualization'):
    """Visualize a specific class"""
    print(f"\n{'='*60}")
    print(f"Visualizing Class {class_id}: {CLASS_NAMES[class_id]}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', 'visualize.py',
        '--config', 'vis_config_reidac9.yaml',
        '--checkpoint', checkpoint,
        '--vis_cls', str(class_id),
        '--nmbr_samples', str(num_samples),
        '--top_traits', str(top_traits),
        '--vis_outdir', output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Class {class_id} ({CLASS_NAMES[class_id]}) completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error visualizing class {class_id}: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Visualize Reidac9 dataset with Prompt_CAM')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to the model checkpoint')
    parser.add_argument('--classes', type=str, default='all',
                        help='Classes to visualize: "all" or comma-separated list (e.g., "0,4,7")')
    parser.add_argument('--nmbr_samples', type=int, default=10,
                        help='Number of samples to visualize per class')
    parser.add_argument('--top_traits', type=int, default=4,
                        help='Number of top traits per sample to visualize')
    parser.add_argument('--vis_outdir', type=str, default='./visualization',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Determine which classes to visualize
    if args.classes.lower() == 'all':
        classes_to_viz = list(range(9))
    else:
        try:
            classes_to_viz = [int(c.strip()) for c in args.classes.split(',')]
            # Validate class IDs
            for c in classes_to_viz:
                if c not in CLASS_NAMES:
                    print(f"Error: Invalid class ID {c}. Must be 0-8")
                    return
        except ValueError:
            print("Error: Invalid class specification. Use 'all' or comma-separated numbers (e.g., '0,4,7')")
            return
    
    print("\n" + "="*60)
    print("Reidac9 Dataset Visualization")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes to visualize: {', '.join([f'{c} ({CLASS_NAMES[c]})' for c in classes_to_viz])}")
    print(f"Samples per class: {args.nmbr_samples}")
    print(f"Top traits: {args.top_traits}")
    print(f"Output directory: {args.vis_outdir}")
    print("="*60 + "\n")
    
    # Visualize each class
    success_count = 0
    for class_id in classes_to_viz:
        if visualize_class(args.checkpoint, class_id, args.nmbr_samples, 
                          args.top_traits, args.vis_outdir):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("Visualization Summary")
    print("="*60)
    print(f"Successfully visualized: {success_count}/{len(classes_to_viz)} classes")
    print(f"Results saved in: {args.vis_outdir}/dino/reidac9/")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
