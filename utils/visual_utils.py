import os
import sys
import torch
import warnings

import numpy as np
import random
import cv2
import shutil


from time import sleep
from random import randint

from PIL import Image

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data.dataset.utils import get_transformation


warnings.filterwarnings("ignore")

def combine_images(path, pred_class,resize_dim=(200,200)):
    images = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.jpg')]
    images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    imgs = [Image.open(image).resize(resize_dim) for image in images]

    widths, heights = zip(*(img.size for img in imgs))

    total_width = sum(widths)
    max_height = max(heights)
    merged_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in imgs:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width
    merged_image.save(path + "/" + "concatenated_prediction_"+str(pred_class)+".jpg")

    for image in images:
        #print(image)
        os.remove(image)

def SuperImposeHeatmap(attention, input_image):
    alpha = 0.5
    
    attention_resized = cv2.resize(attention, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Check if the attention map is already normalized
    min_val = attention_resized.min()
    max_val = attention_resized.max()
    attention_normalized = (attention_resized - min_val) / (max_val - min_val)


    # Apply Gaussian blur for smoothing
    attention_normalized = cv2.GaussianBlur(attention_normalized, (9, 9), 0)

    # Convert to heatmap
    heatmap = (attention_normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on the original image
    result = (input_image * alpha + heatmap * (1 - alpha)).astype(np.uint8)
    return result

def create_overlay_images(X,patch_size,attentions,output_folder):
    w_featmap = X.shape[-1] // patch_size
    attentions =attentions[0]

    nh = attentions.shape[0]
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    image = X[0].detach().cpu().numpy().transpose(1, 2, 0)  # Shape (H, W, C)
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_folder, "0.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    for head in range(nh):
        attention_map = attentions[head].reshape(w_featmap, w_featmap).detach().cpu().numpy()
        result_image = SuperImposeHeatmap(attention_map, image)

        # Save the overlayed image
        save_path = os.path.join(output_folder, f"{head+1}.jpg")
        cv2.imwrite(save_path, result_image)


def prune_and_plot_ranked_heads(model,inputs,target, params):
    if params.top_traits<1 and params.top_traits>model.num_heads:
        raise notImplementedError("top_traits must be greater than 0 and less than the number of heads")

    remaining_head_list = list(range(model.num_heads))
    pruned_head_index = None
    blur_head_lst = []
    blur_head_probs = []

    # Determine the ranking of heads by iteratively finding the head that, when blurred,
    # gives the highest probability for the target.
    while len(remaining_head_list) > 0:
        highest_score=-1e8
        remaining_head_scores= []

        for head_idx in remaining_head_list:
            output,_ = model(inputs,
                            blur_head_lst=blur_head_lst+[head_idx],
                            target_cls=target)
            
            probabilities = torch.softmax(output.squeeze(-1), dim=-1)

            remaining_head_scores.append(probabilities[0,target].item())

            if remaining_head_scores[-1] > highest_score:
                highest_score=remaining_head_scores[-1] 
                pruned_head_index=head_idx

        if pruned_head_index is not None:
            blur_head_lst.append(pruned_head_index)
            remaining_head_list.remove(pruned_head_index)
            blur_head_probs.append(highest_score)  

    #### Convert the image for overlaying the attention maps
    image = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)  # Shape (H, W, C)
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    w_featmap = inputs.shape[-1] // model.patch_size

    ### Go through all the attention maps and overlay them on the image
    _,attn_maps = model(inputs)
    attn_maps = attn_maps[:, :, target, (params.vpt_num+1):]
    overlayed_attn_maps = []

    for head in range(model.num_heads):
        attention_map = attn_maps[0,head].reshape(w_featmap, w_featmap).detach().cpu().numpy()
        overlayed_attn_maps.append(cv2.cvtColor(SuperImposeHeatmap(attention_map, image),cv2.COLOR_BGR2RGB))

    #### Create captions for all the plot
    captions = ['input image']
    for head in range(model.num_heads):
        captions.append(f'rank {head+1}')

    print(f'Head # (from most important to least important):{blur_head_lst[::-1]}')
    #### Create plot with overlayed attention maps
    for bl_idx in range(1,len(blur_head_lst)+1):
        if bl_idx == params.top_traits+1:
            break
        current_blr_head_lst = blur_head_lst[:-bl_idx]
        remaining_head_list = remaining_head_list+ [blur_head_lst[-bl_idx]]

    current_images = [image]
    for r_idx in remaining_head_list:
        current_images+=[overlayed_attn_maps[r_idx]]

    # Create a grid of images
    grid_size = (1,len(current_images))
    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(current_images[i])
        ax.axis('off')
        ax.set_title(captions[i])

    plt.tight_layout()
    plt.show()
        

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    transformation = get_transformation(mode='test')
    transformed_image = transformation(image)
    return torch.unsqueeze(transformed_image, 0)