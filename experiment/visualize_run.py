import os
from experiment.build_model import get_model
from experiment.build_loader import get_loader
from timm.utils import get_outdir,accuracy
from utils.log_utils import logging_env_setup
from utils.misc import AverageMeter
import torch
import numpy as np
from utils.setup_logging import get_logger

from utils.visual_utils import combine_images,create_overlay_images


logger = get_logger("Prompt_CAM")


def basic_vis(params):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dataset_name = params.data.split("-")[0]
    top_traits = f"top_traits_{params.top_traits}"
    output_dir = os.path.join(params.vis_outdir, params.model, dataset_name, "class_"+str(params.vis_cls),top_traits)
    params.output_dir = get_outdir(output_dir)
    logging_env_setup(params)

    
    logger.info(f'Start loading test data: {dataset_name}')
    _, _, test_loader = get_loader(params, logger)

    logger.info(f'Start loading model: {params.model}')
    model, _ , _ = get_model(params)
    model.load_state_dict(torch.load(params.checkpoint)['model_state_dict'])
    logger.info (f'Model loaded from {params.checkpoint}')

    top1_m = AverageMeter()

    model.eval()


    params.test_batch_size= 1
    _, _, test_loader = get_loader(params, logger)


    smpl_count = 0
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(test_loader):
            # move data to device
            samples = samples.to(params.device, non_blocking=True)  # (batchsize, channel, height, width)
            targets = targets.to(params.device, non_blocking=True)  # (batchsize, )

            if targets[0].item() == params.vis_cls:
                smpl_count += 1

                outputs, attn_map = model(samples)
                predicted_class = torch.argmax(outputs, dim=1).item() 

                if predicted_class == targets[0].item():
                    logger.info(f"Predicted class: {predicted_class}, Target class: {targets[0].item()}")
                    prune_attn_heads(model,samples,targets, predicted_class,smpl_count, params)
                else:
                    attn_map = attn_map[:, :, targets[0].item(), (params.vpt_num+1):]
                    create_overlay_images(samples,
                                        model.patch_size,
                                        attn_map,
                                        f'{params.output_dir}/img_{smpl_count}')

                    combine_images(path=f'{params.output_dir}/img_{smpl_count}', pred_class=predicted_class)


                if smpl_count == params.nmbr_samples:
                    break




    #TODO: ADD Later
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(test_loader):
            # move data to device
            samples = samples.to(params.device, non_blocking=True)  # (batchsize, 2048)
            targets = targets.to(params.device, non_blocking=True)  # (batchsize, )

            outputs,_ = model(samples)
            acc1,_= accuracy(outputs.squeeze(-1), targets, topk=(1,5))
            top1_m.update(acc1.item(), samples.shape[0])

            del outputs, acc1

        logger.info("Evaluate: average top1: {:.2f}".format(top1_m.avg))


def prune_attn_heads(model,inputs,target, prediction,smpl_count, params):
    remaining_head_list = list(range(model.num_heads))
    pruned_head_index = None
    blur_head_lst = []

    while len(remaining_head_list) > 0 and len(remaining_head_list) > params.top_traits:
        highest_score=-1e8
        remaining_head_scores= []

        for head_idx in remaining_head_list:
            output,_ = model(inputs,
                            blur_head_lst=blur_head_lst+[head_idx],
                            target_cls=prediction)
            
            probabilities = torch.softmax(output.squeeze(-1), dim=-1)

            remaining_head_scores.append(probabilities[0,prediction].item())

            if remaining_head_scores[-1] > highest_score:
                highest_score=remaining_head_scores[-1] 
                pruned_head_index=head_idx

        if pruned_head_index is not None:
            blur_head_lst.append(pruned_head_index)
            remaining_head_list.remove(pruned_head_index)
            print(f'best head to prune is {pruned_head_index+1} with score {highest_score}')    

    sorted_remaining_heads = [head for _, head in sorted(zip(remaining_head_scores, remaining_head_list))]

    _,attn_map=model(inputs,
                    blur_head_lst=blur_head_lst,
                    target_cls=prediction)
    attn_map = attn_map[:, sorted_remaining_heads, prediction, (params.vpt_num+1):]
    create_overlay_images(inputs,
                        model.patch_size,
                        attn_map,
                        f'{params.output_dir}/img_{smpl_count}')

    combine_images(path=f'{params.output_dir}/img_{smpl_count}', pred_class=prediction)




