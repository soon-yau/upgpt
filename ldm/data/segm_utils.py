import os, sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from random import choice
from abc import abstractmethod
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from einops import rearrange
from glob import glob
import random
import json
from tqdm import tqdm

from collections import OrderedDict
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map 

import numpy as np

class Segmenter:
    def __init__(self, 
                 label_dict: dict, 
                 segm_groups: dict, 
                 image_transform,
                 feat_transform=None,
                 device=None):
        self.image_transform = image_transform
        self.feat_transform = feat_transform
        self.device = device
        self.label_dict = label_dict
        self.label2id = dict(zip(self.label_dict.values(), self.label_dict.keys()))
        self.segm_groups = segm_groups
        self.segm_id_groups = OrderedDict()
        for k, v in self.segm_groups.items():
            self.segm_id_groups[k] = [self.label2id[l] for l in v]
                
    def get_mask(self, segm, mask_val, default_value=1.0):
        mask = np.full(segm.shape, default_value, dtype=np.float32)
        for label, value in mask_val.items():
            mask[segm==self.label2id[label]] = value
        return mask       
    
    
    def get_binary_mask(self, segm, mask_ids):
        mask = np.full(segm.shape, False)
        for mask_id in mask_ids:
            mask |= segm == mask_id    
        return mask        
        
    def get_mask_range(self, mask, margin):
        height, width = mask.shape
        left = 0
        right = width
        top = 0
        bottom = height
        
        vertical = torch.sum(mask.to(torch.float32), dim=0)
        for w in range(width):
            if vertical[w] > 0.1:
                left = w
                break

        for w in range(width-1, -1, -1):
            if vertical[w] > 0.1:
                right = w
                break
        left = max(0, left-margin)
        right = min(width, right+margin)

        horizontal = torch.sum(mask.to(torch.float32), dim=1)

        for h in range(height):
            if horizontal[h] > 0.1:
                top = h
                break

        for h in range(height-1, -1, -1):
            if horizontal[h] > 0.1:
                bottom = h
                break

        top = max(0, top-margin)
        bottom = min(height, bottom+margin)

        return {'left':left, 'right':right, 'top':top, 'bottom':bottom}

    def crop(self, 
             input_image: torch.tensor,
             mask: np.array, 
             margin=0,
             is_background=False,
             mask_background=False,
             name=None):
        
        image = torch.clone(input_image).detach()
        mask_range = self.get_mask_range(mask, margin)
        
        if is_background: # fill the mask with average background color
            new_images = []
            for i in range(3):
                mean_color = torch.masked_select(image[i], mask==True).mean()
                new_images.append(image[i].masked_fill(mask==False, mean_color))

            cropped = torch.stack(new_images)
        else:    
            cropped = image * mask if mask_background else image

            cropped = cropped[:,mask_range['top']:mask_range['bottom'], 
                                mask_range['left']:mask_range['right']]
            if name=='face' and (mask_range['bottom']-mask_range['top'])>128:
                return torch.zeros((3, 224,224))
            if cropped.sum() > 0:
                _, h, w = cropped.shape
                pad = (h - w)//2

                if pad > 0:
                    padding  = (pad, pad, 0, 0)
                    cropped = torch.nn.ZeroPad2d(padding)(cropped)
                elif pad < 0:
                    padding  = (0, 0, -pad, -pad)
                    cropped = torch.nn.ZeroPad2d(padding)(cropped)                
            else:
                return torch.zeros((3, 224,224))
        cropped = self.image_transform(cropped)
        if self.feat_transform:
            cropped = self.feat_transform(cropped)
        return cropped
    
    def forward(self, image, segm:np.array):
        image = T.ToTensor()(image)
        if self.device:
            image = image.to(self.device)
        cropped_images =  OrderedDict()
        for name, segm_group in self.segm_id_groups.items():
            mask = torch.from_numpy(self.get_binary_mask(segm, segm_group)).to(self.device)
            
            cropped = self.crop(image, mask, 
                                margin=0,#margins[name],
                                is_background=name=='background',
                                name=name,
                                mask_background=name!='face')
            cropped_images[name] = cropped
            
        return cropped_images
    
class LipSegmenter(Segmenter):
    def __init__(self, feat_transform=None):
        
        label_names = ['background', 'hat', 'hair', 'glove', 'eyeglass', 'top', 'dress', 'coat',
                  'socks', 'pants', 'jumpsuits', 'scarf', 'skirt', 'face', 'left-arm', 'right-arm',
                  'left-leg', 'right-leg', 'left-shoe', 'right-shoe']
        
        label_dict = dict(zip([i for i in range(len(label_names))], label_names))
        
        segm_groups = OrderedDict({
            'face':['eyeglass','face'],
            'background':['background'],
            'hair': ['hair'],
            'headwear': ['hat'],
            'top':[ 'top', 'dress','jumpsuits','scarf'],
            'bottom':['skirt','dress','pants','jumpsuits'],
            'shoes':['left-shoe','right-shoe','socks'],
            'outer': ['coat'],
            #'arms':['left-arm', 'right-arm']
            })

        image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=224),
            T.CenterCrop(size=(224, 224))])
        
        super().__init__(label_dict, segm_groups, image_transform, feat_transform)

            
clip_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                            std=(0.26862954, 0.26130258, 0.27577711))
])


class DeepfashionMMSegmenter(Segmenter):
    def __init__(self, feat_transform=None, **kwargs):
        
        label_dict={
            0: 'background',
            1: 'top',
            2: 'outer',
            3: 'skirt',
            4: 'dress',
            5: 'pants',
            6: 'leggings',
            7: 'headwear',
            8: 'eyeglass',
            9: 'neckwear',
            10: 'belt',
            11: 'footwear',
            12: 'bag',
            13: 'hair',
            14: 'face',
            15: 'skin',
            16: 'ring',
            17: 'wrist wearing',
            18: 'socks',
            19: 'gloves',
            20: 'necklace',
            21: 'rompers',
            22: 'earrings',
            23: 'tie'}
        
        segm_groups = {
            'face':['eyeglass','face'],
            'background':['background'],
            'hair': ['hair'],
            'headwear': ['hat'],
            'top':[ 'top', 'dress','jumpsuits','scarf'],
            'bottom':['skirt','dress','pants','jumpsuits'],
            'shoes':['left-shoe','right-shoe','socks'],
            'outer': ['coat'],
            'arms':['left-arm', 'right-arm']
            }

        image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=224),
            T.CenterCrop(size=(224, 224))])
        
        super().__init__(label_dict, segm_groups, image_transform, feat_transform, **kwargs)
