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

segm_groups = {
    'face':['eyeglass','face'],
    'hair': ['hair'],
    'headwear': ['headwear'],
    'background':['background'],
    'top':[ 'top','rompers','dress'],
    'bottom':['skirt','dress','leggings','pants', 'belt'],
    'shoes':['footwear','socks'],
    'outer': ['outer'],
    'accesories':['bag']
    }

class DeepfashionMMSegment:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.label_dict={
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
        
        self.label2id = dict(zip(self.label_dict.values(), self.label_dict.keys()))
        
        self.segm_id_groups = OrderedDict()
        for k, v in segm_groups.items():
            self.segm_id_groups[k] = [self.label2id[l] for l in v]
                
        self.clip_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=224),
            T.CenterCrop(size=(224, 224)),
            #T.ToTensor(),
            #T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
    def get_mask(self, segm, mask_ids):
        mask = np.full(segm.shape, False)
        for mask_id in mask_ids:
            mask |= segm== mask_id    
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

    def crop(self, input_image, # torch tensor
             mask, 
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
                return None
        return self.clip_transform(cropped)

    def forward(self, image, segm):

        image = T.ToTensor()(image).to(self.device)
        cropped_images =  OrderedDict()
        for name, segm_group in self.segm_id_groups.items():
            mask = torch.from_numpy(self.get_mask(segm, segm_group)).to(self.device)
            
            cropped = self.crop(image, mask, 
                                margin=0,#margins[name],
                                is_background=name=='background',
                                name=name,
                                mask_background=name!='face')
            cropped_images[name] = cropped
            
        return cropped_images
    
segmentor = DeepfashionMMSegment(device='cuda:0')

image_root = '/home/soon/datasets/deepfashion_inshop/img_highres/'
segm_root = '/home/soon/datasets/deepfashion_inshop/segm/'
dst_root = '/home/soon/datasets/deepfashion_inshop/styles/'
segm_files = glob(os.path.join(segm_root,'**/*_segm.png'),recursive=True)

for segm_file in tqdm(segm_files[:]):
#def extract(segm_file):
    image_file = segm_file.replace('_segm.png','.jpg').replace(segm_root, image_root)
    image = np.array(Image.open(image_file))
    segm = np.array(Image.open(segm_file))
    cropped = segmentor.forward(image, segm)
    file_id = segm_file.replace('_segm.png','')
    path, fname = os.path.split(file_id)
    dst_dir = os.path.join(path, fname.replace('_','/',1)).replace(segm_root, dst_root)
    os.makedirs(dst_dir, exist_ok=True)
    for k, v in cropped.items():
        if v!= None:
            crop_file = os.path.join(dst_dir, k+'.jpg')
            v.save(crop_file)

#with Pool(os.cpu_count()-1) as p:
#with Pool(8) as p:
#    p.map(extract, segm_files)
#process_map(extract, segm_files, max_workers=8)
print(f'Processed {len(segm_files)} files.')