import os
import os.path
import torchvision as tv

import torch.utils.data as data
from PIL import Image
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def add_samples(sample_list, list_path, root):
    with open(list_path, 'r') as f:
        for line in f:
            img_name = line.rsplit(' ', 1)[0]
            label = int(line.rsplit(' ', 1)[1])
            sample_list.append((os.path.join(root, img_name), label))

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, name, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.name = name
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)

def get_transformation(mode, mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD):
    if mode == 'train':
        return tv.transforms.Compose([
            tv.transforms.Resize((240,240)),
            tv.transforms.RandomCrop((224,224)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std),
        ])
    elif mode == 'val' or mode == 'test':
        return tv.transforms.Compose([
            tv.transforms.Resize((224,224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError("Invalid mode. Use 'train' or 'val' or 'test'.")

def create_annotation_file(data_path, mode, output_file):
    """
    Creates a file listing images and their corresponding labels based on the directory structure.

    Args:
        data_path: The root directory containing the dataset (e.g., dataset_name/).
        output_file: The path to the file where the image list will be written.
    """

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)  # Get the directory part of the path
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = []
    label_map = {}  # Store species names and corresponding numerical labels

    # Assign numerical labels to species names in the order they're encountered
    label_counter = 0

    for split in mode:  # Iterate through train and val splits
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            continue # Skip if split does not exist

        for species_dir in sorted(os.listdir(split_path)):  # Iterate through species directories
            if not os.path.isdir(os.path.join(split_path, species_dir)):
                continue # Skip if not a directory
            species_name = species_dir.split('.', 1)[1] if '.' in species_dir else species_dir # extract species name
            if species_name not in label_map:
                label_map[species_name] = label_counter
                label_counter += 1
            label = label_map[species_name]

            image_dir = os.path.join(split_path, species_dir)
            for image_file in os.listdir(image_dir):  # Iterate through image files
                if os.path.isfile(os.path.join(image_dir, image_file)): # ensure it is a file
                  image_path = os.path.join(split, species_dir, image_file)  # Relative path from dataset root
                  image_list.append(f"{image_path} {label}")

    # Write the image list to the output file
    with open(output_file, "w") as f:
        f.write("\n".join(image_list))
