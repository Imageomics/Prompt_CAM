"""
Prepare CUB-200-2011 dataset into the Prompt-CAM directory structure.

Download the dataset from:
    https://www.vision.caltech.edu/datasets/cub_200_2011/

After extracting the tarball you should have:
    CUB_200_2011/
    ├── images/
    ├── images.txt
    └── train_test_split.txt

Run:
    python data/prepare_cub.py --cub_dir /path/to/CUB_200_2011 --out_dir /path/to/data/images/cub
"""

import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cub_dir",
        required=True,
        help="Path to the extracted CUB_200_2011 folder.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Destination folder (e.g. data/images/cub). Created if absent.",
    )
    return parser.parse_args()


def read_split(cub_dir):
    """Return dict {image_id: 'train' or 'val'}."""
    split_file = os.path.join(cub_dir, "train_test_split.txt")
    split = {}
    with open(split_file) as f:
        for line in f:
            img_id, is_train = line.strip().split()
            split[img_id] = "train" if is_train == "1" else "val"
    return split


def read_images(cub_dir):
    """Return dict {image_id: relative_path_from_images_dir}."""
    images_file = os.path.join(cub_dir, "images.txt")
    images = {}
    with open(images_file) as f:
        for line in f:
            img_id, img_path = line.strip().split(maxsplit=1)
            images[img_id] = img_path
    return images


def main():
    args = parse_args()
    cub_dir = args.cub_dir
    out_dir = args.out_dir

    split = read_split(cub_dir)
    images = read_images(cub_dir)

    images_src = os.path.join(cub_dir, "images")

    for img_id, img_rel_path in images.items():
        subset = split[img_id]
        # img_rel_path is like "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
        class_folder = img_rel_path.split("/")[0]
        filename = img_rel_path.split("/")[1]

        dst_dir = os.path.join(out_dir, subset, class_folder)
        os.makedirs(dst_dir, exist_ok=True)

        src = os.path.join(images_src, img_rel_path)
        dst = os.path.join(dst_dir, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    train_classes = len(os.listdir(os.path.join(out_dir, "train")))
    val_classes = len(os.listdir(os.path.join(out_dir, "val")))
    print(f"Done. train: {train_classes} classes | val: {val_classes} classes")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
