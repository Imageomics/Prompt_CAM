import os
import pprint
import sys
from collections import defaultdict
import torch
from tabulate import tabulate

from utils.distributed import get_rank, get_world_size
from utils.setup_logging import setup_logging


def get_env_module():
    var_name = "ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_torch_env() -> str:
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_env_info():
    data = []
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(get_env_module())
    data.append(("PyTorch", torch.__version__))
    data.append(("PyTorch Debug Build", torch.version.debug))

    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        data.append(("CUDA ID", os.environ["CUDA_VISIBLE_DEVICES"]))
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def logging_env_setup(params) -> None:
    logger = setup_logging(
        params.gpu_num, get_world_size(), params.output_dir, name="Prompt_CAM")

    # Log basic information about environment, cmdline arguments, and config
    rank = get_rank()
    logger.info(
        f"Rank of current process: {rank}. World size: {get_world_size()}")
    logger.info("Environment info:\n" + collect_env_info())


    # Show the config
    logger.info("Training with config:")
    logger.info(pprint.pformat(params))

def log_model_info(model, logger, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    model_grad_params_no_head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'head' not in n)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}\t Gradient Parameters No Head: {2}".format(
        model_total_params, model_grad_params, model_grad_params_no_head))
    logger.info(f"total tuned percent:{(model_grad_params/model_total_params*100):.2f} %")
    logger.info(f"total tuned percent no head:{(model_grad_params_no_head / model_total_params * 100):.2f} %")

    # Print the names of parameters that require gradient updates
    fine_tuned_params = [n for n, p in model.named_parameters() if p.requires_grad]
    
    logger.info("Fine-tuned Parameters:")
    for param_name in fine_tuned_params:
        logger.info(param_name)
        
    return model_grad_params_no_head

