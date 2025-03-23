from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
from timm.data.transforms import str_to_interp_mode
from data.dataset.utils import add_samples, create_annotation_file, get_transformation
import os

class PetDataset(ImageFolder):
    def __init__(self, root, data_list, transform=None):
        self.data_root = root
        self.loader = default_loader
        self.transform = transform
        self.target_transform = None
        self.samples = []

        add_samples(self.samples, data_list, root)


def get_pet(params, mode='trainval_combined'):
    params.class_num = 37
    mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

    transform_train = get_transformation('train', mean, std)
    transform_val = get_transformation('val', mean, std)
    
    if mode == 'trainval_combined':
        train_data_list = f'data/annotations/pet/{params.data}_combine.txt'

        if not os.path.exists(train_data_list):
            create_annotation_file(params.data_path, ['train'],train_data_list)
        return PetDataset(params.data_path, train_data_list, transform_train)

    elif mode == 'test':
        test_data_list = f'data/annotations/pet/test.txt'
        if not os.path.exists(test_data_list):
            create_annotation_file(params.data_path,['val'],test_data_list)
        return PetDataset(params.data_path, test_data_list, transform_val)
    else:
        raise NotImplementedError


