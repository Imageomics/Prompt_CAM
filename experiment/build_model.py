from tkinter.constants import RAISED

import timm
import torch
from model.vision_transformer import VisionTransformerPETL
from utils.log_utils import log_model_info
from timm.data import resolve_data_config
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")

TUNE_MODULES = ['vpt']
def get_model(params,visualize=False):
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {params.device}")

    model = get_base_model(params,visualize=visualize)

    ##########
    tune_parameters = []
    if params.debug:
        logger.info("Trainable params:")

    for name, parameter in model.named_parameters():
        if any(m in name for m in TUNE_MODULES):
            parameter.requires_grad = True
            tune_parameters.append(parameter)
            if params.debug:
                logger.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
        else:
            parameter.requires_grad = False

    model_grad_params_no_head = log_model_info(model, logger)

    if not visualize:
      model = model.cuda(device=params.device)
    return model, tune_parameters, model_grad_params_no_head


def get_base_model(params,visualize=False):
    if params.pretrained_weights == "vit_base_patch16_224_in21k":
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_224_in21k_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False, params=params)
        if not visualize:
            model.load_pretrained(
            'pretrained_weights/ViT-B_16_in21k.npz')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_mae":
        model = timm.create_model("vit_base_patch16_224_in21k_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        if not visualize:
            model.load_pretrained(
            'pretrained_weights/mae_pretrain_vit_base.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_patch14_dinov2":
        params.patch_size = 14
        model = timm.create_model("vit_base_patch14_dinov2_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        if not visualize:
            model.load_pretrained(
            'pretrained_weights/dinov2_vitb14_pretrain.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == "vit_base_patch16_dino":
        model = timm.create_model("vit_base_patch16_dino_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        if not visualize:
            model.load_pretrained(
            'pretrained_weights/dino_vitbase16_pretrain.pth')
        model.reset_classifier(params.class_num)
    elif params.pretrained_weights == 'vit_base_patch16_clip_224':
        params.patch_size = 16
        model = timm.create_model("vit_base_patch16_clip_224_petl", drop_path_rate=params.drop_path_rate,
                                  pretrained=False,
                                  params=params)
        if not visualize:
            model.load_pretrained(
            'pretrained_weights/ViT-B_16_clip.bin')

        fc = init_imagenet_clip(params.device)
        proj = get_clip_proj(params.device)
        model.head = torch.nn.Sequential(*[proj, fc])
    else:
        raise NotImplementedError

    # data_config = resolve_data_config(vars(params), model=model, verbose=False)

    return model
