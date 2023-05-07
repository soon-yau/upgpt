import os
import sys
#sys.path.append(os.path.join(os.getcwd(), '..'))
from einops import rearrange
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from ldm.util import instantiate_from_config


from copy import deepcopy
import argparse, os, sys, glob
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pathlib import Path


from skimage.metrics import structural_similarity as ssim
from ldm.data.deepfashion_inshop import DeepFashionSample
from ldm.data.segm_utils import LipSegmenter

import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import random
import re

style_names = ['face', 'hair', 'headwear', 'background', 'top', 'outer', 'bottom', 'shoes', 'accesories']

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model


def draw_styles(style_batch):
    
    #style_names = ['face', 'hair', 'headwear', 'background', 'top', 'outer', 'bottom', 'shoes', 'accesories']

    denorm = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],  std = [ 1/0.226862954, 1/0.26130258, 1/0.27577711 ]),
                         T.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073], std = [ 1., 1., 1. ]),      ])
    rows, cols = 2, 4
    fig, axs = plt.subplots(rows, cols)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    for i, (name, style) in enumerate(zip(style_names[:-1], style_batch[:-1])):
        row = i//cols
        col = i%cols
        axs[row, col].imshow(T.ToPILImage()(denorm(style)))
        axs[row, col].set_title(name)
        axs[row, col].axis('off')
    plt.show()
            

def convert_fname(long_name):
    '''
    convert fashionWOMENBlouses_Shirtsid0000311501_7additional___fashionWOMENBlouses_Shirtsid0000311501_2side'
    to 'WOMEN/Blouses_Shirts/id_00003115/01_7_additional',
    'WOMEN/Blouses_Shirts/id_00003115/01_2_side'
    '''    
    gender = 'MEN' if long_name[7:10]  == 'MEN' else 'WOMEN'

    input_list = long_name.replace('fashion','').split('___')
    
    # Define a regular expression pattern to match the relevant parts of each input string
    if gender == 'MEN':
        pattern = r'MEN(\w+)id(\d+)_(\d)(\w+)'
    else:
        pattern = r'WOMEN(\w+)id(\d+)_(\d)(\w+)'
    # Use a list comprehension to extract the matching substrings from each input string, and format them into the desired output format
    output_list = [f'{gender}/{category}/id_{id_num[:8]}/{id_num[8:]}_{view_num}_{view_desc}' for (category, id_num, view_num, view_desc) in re.findall(pattern, ' '.join(input_list))]

    # Print the resulting list of formatted strings
    return output_list

clip_transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))
                          ])

def get_empty_style():
    return clip_transform(np.zeros((224,224,3)))



#mix_style(dst_batch['styles'], text_prompt).shape

def get_coord(batch_mask):
    mask = batch_mask[0].cpu().numpy()
    mask[mask==-1] = 0
    x = np.nonzero(np.mean(mask,1))[0]
    xmin, xmax = x[0], x[-1]
    y = np.nonzero(np.mean(mask,0))[0]
    ymin, ymax = y[0], y[-1]

    return np.array([xmin, xmax, ymin, ymax])

def get_mask(mask, coord):
    device = mask.device
    xmin, xmax, ymin, ymax = coord
    new_mask = np.ones_like(mask.cpu().numpy())*(-1)
    new_mask[0,xmin:xmax+1, ymin:ymax+1] = -0.99215686
    #return new_mask
    return torch.tensor(new_mask).to(device)

def interp_mask(src_mask, dst_mask, alpha):    
    coord_1 = get_coord(src_mask)
    coord_2 = get_coord(dst_mask)

    coord = (alpha * coord_1 + (1 - alpha) * coord_2).astype(np.int32)

    new_mask = get_mask(src_mask, coord)
    return new_mask


class InferenceModel:
    def __init__(self, config, ckpt, device):
        self.device = device
        text_encoder_config = {'target': 'ldm.modules.encoders.modules.FrozenCLIPTextEmbedder', 
                       'params': {'normalize': False}}
        self.clip_text_encoder = instantiate_from_config(text_encoder_config).to(device)
        self.clip_image_encoder = instantiate_from_config(config['model']['params']['extra_cond_stages']['style_cond']).to(device)
        config['model']['params']['extra_cond_stages']['style_cond']['target'] = 'ldm.modules.poses.poses.DummyModel'
        config['model']['params']['first_stage_config']['params']['ckpt_path'] = None        
        self.model = load_model_from_config(config, f"{ckpt}").to(device)

    def create_batch(self, batch, repeat=1):
        for k, v in batch.items():
            if type(v)==torch.Tensor:
                temp = batch[k].unsqueeze(0)
                repeat_list = [1]*len(temp.shape)
                repeat_list[0] = repeat
                batch[k] = temp.repeat(repeat_list).to(self.device)                        
            else:
                batch[k] = [batch[k]]*repeat
        return batch

    def generate(self, batch, steps=200, repeat=1, use_ema=True):
        
        with torch.no_grad():
            images = self.model.log_images(batch, ddim_steps=steps, use_ema=use_ema,
                                    unconditional_guidance_scale=3.,
                                    unconditional_guidance_label=[""])
            
        for k in images:
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1., 1.)
            images[k] = rearrange(images[k].numpy(),'b c h w -> b h w c') *0.5 + 0.5
        return images


    def mix_style(self, s, w, mask=[]):
        style2id = dict(zip(style_names,[x for x in range(len(style_names))]))
        text_dict = dict(zip(style_names, ['' for _ in range(len(style_names))]))
        
        for m in mask:
            s[style2id[m]] = get_empty_style()
            
        with torch.no_grad():        
            for k, v in w.items():
                text_dict[k] = v
            texts = list(text_dict.values())
            text_emb = self.clip_text_encoder([texts])
            
            image_emb = self.clip_image_encoder(s.unsqueeze(0).to(self.device))

            for i, text in enumerate(texts):
                if text != '':                
                    image_emb[0,i] = text_emb[0,i]
        return image_emb.squeeze(0)