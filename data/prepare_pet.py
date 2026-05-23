"""
Prepare Oxford-IIIT Pet dataset into the Prompt-CAM directory structure.

Download the dataset from:
    https://www.robots.ox.ac.uk/~vgg/data/pets/

You need two archives:
    images.tar.gz        -> extract to get a folder of .jpg images
    annotations.tar.gz   -> extract to get trainval.txt and test.txt

After extracting you should have:
    <some_dir>/
    ├── images/               (all .jpg images, e.g. Abyssinian_1.jpg)
    └── annotations/
        ├── trainval.txt
        └── test.txt

Each annotation line has the format:
    <image_name> <class_id> <species_id> <breed_id>
where class_id is 1-indexed and corresponds to alphabetical order of breed names.

Run:
    python data/prepare_pet.py \
        --images_dir /path/to/images \
        --annotations_dir /path/to/annotations \
        --out_dir /path/to/data/images/pet
"""

import argparse
import os
import re
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Path to the extracted images/ folder containing all .jpg files.",
    )
    parser.add_argument(
        "--annotations_dir",
        required=True,
        help="Path to the extracted annotations/ folder containing trainval.txt and test.txt.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Destination folder (e.g. data/images/pet). Created if absent.",
    )
    return parser.parse_args()


def class_name_from_image(image_name):
    """'Abyssinian_1' -> 'Abyssinian', 'English_Cocker_Spaniel_12' -> 'English_Cocker_Spaniel'."""
    return re.sub(r"_\d+$", "", image_name)


def parse_annotation_file(ann_file):
    """Return list of (image_name, class_id) skipping comment lines."""
    entries = []
    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_name = parts[0]
            class_id = int(parts[1])
            entries.append((image_name, class_id))
    return entries


def main():
    args = parse_args()
    images_dir = args.images_dir
    annotations_dir = args.annotations_dir
    out_dir = args.out_dir

    trainval_file = os.path.join(annotations_dir, "trainval.txt")
    test_file = os.path.join(annotations_dir, "test.txt")

    splits = [
        (trainval_file, "train"),
        (test_file, "val"),
    ]

    for ann_file, subset in splits:
        if not os.path.exists(ann_file):
            print(f"Warning: {ann_file} not found, skipping {subset} split.")
            continue

        entries = parse_annotation_file(ann_file)
        for image_name, class_id in entries:
            breed = class_name_from_image(image_name)
            # Zero-pad class_id to three digits to match the NNN.ClassName convention
            class_folder = f"{class_id:03d}.{breed}"

            src = os.path.join(images_dir, image_name + ".jpg")
            if not os.path.exists(src):
                print(f"Warning: image not found: {src}")
                continue

            dst_dir = os.path.join(out_dir, subset, class_folder)
            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(dst_dir, image_name + ".jpg")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    for subset in ("train", "val"):
        subset_path = os.path.join(out_dir, subset)
        if os.path.exists(subset_path):
            n = len(os.listdir(subset_path))
            print(f"{subset}: {n} classes")

    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
