from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
from timm.data.transforms import str_to_interp_mode
from data.dataset.utils import add_samples, create_annotation_file, get_transformation
import os

class CubDataset(ImageFolder): ## For new data ## Change this::  the name of the class to match the dataset name
    def __init__(self, root, data_list, transform=None):
        self.data_root = root
        self.loader = default_loader
        self.transform = transform
        self.target_transform = None
        self.samples = []

        add_samples(self.samples, data_list, root)


def get_cub(params, mode='trainval_combined'): ## For new data ## Change this::  the name of the dataset name
    params.class_num = 200  ## For new data ## Change this::  the number of classes in the dataset
    mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

    transform_train = get_transformation('train', mean, std)
    transform_val = get_transformation('val', mean, std)
    
    if mode == 'trainval_combined':
        train_data_list = f'data/annotations/cub/{params.data}_combine.txt' ## For new data ## Change this::  the name of the data in the path

        if not os.path.exists(train_data_list):
            create_annotation_file(params.data_path, ['train'],train_data_list)
        return CubDataset(params.data_path, train_data_list, transform_train)  ## For new data ## Change this::  the class to call

    elif mode == 'test':
        test_data_list = f'data/annotations/cub/test.txt'  ## For new data ## Change this::  the name of the data in the path
        if not os.path.exists(test_data_list):
            create_annotation_file(params.data_path,['val'],test_data_list)
        return CubDataset(params.data_path, test_data_list, transform_val) ## For new data ## Change this::  the class to call
    else:
        raise NotImplementedError


