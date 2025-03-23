import torch
import numpy as np
import random
import time
import yaml
from dotwiz import DotWiz

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def method_name(params):
    name = ''
    if params.train_type == 'prompt_cam':
        name += 'pcam_'
        name += params.train_type + '_'
        name += str(params.vpt_num) + '_'
    elif params.vpt_mode:
        name += 'vpt_'
        name += params.vpt_mode + '_'
        name += str(params.vpt_num) + '_'
        name += str(params.vpt_layer) + '_'
    #####if nothing, linear
    if name == '':
        name += 'linear' + '_'
    name += params.optimizer
    return name


def set_seed(random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def throughput(model,img_size=224,bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size=x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)

def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            return DotWiz(yaml.load(stream, Loader=yaml.FullLoader))
        except yaml.YAMLError as exc:
            print(exc)

def override_args_with_yaml(args, yaml_config):
    """Override argparse args with values from YAML if they exist."""
    for key, value in yaml_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

def load_vis_args_with_yaml(args,yaml_config_path,checkpoint_path):
    """Create args with yaml for notebook"""
    yaml_config = load_yaml(yaml_config_path)
    for key, value in yaml_config.items():
        setattr(args, key, value)

    set_seed(args.random_seed)
    args.checkpoint = checkpoint_path
    args.test_batch_size= 1
    return args
    
class EarlyStop:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_metrics = None

    def early_stop(self, eval_metrics):
        '''

        :param val_acc:
        :return: bool(if early stop), bool(if save model)
        '''
        if self.max_metrics is None:
            self.max_metrics = eval_metrics
        if eval_metrics['top1'] > self.max_metrics['top1']:
            self.max_metrics = eval_metrics
            self.counter = 0
            return False, True
        elif eval_metrics['top1'] < (self.max_metrics['top1'] - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        return False, False
