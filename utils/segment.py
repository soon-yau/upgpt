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

from collections import OrderedDict

segm_table = {
    (0, 0, 0)      : 0, # background 
    (255, 250, 250): 1, # top     
    (250, 235, 215): 3, #    skirt     
    (70, 130, 180) : 6, #   leggings   
    (16, 78, 139)  : 14, # face    
    (255, 250, 205): 4, #     dress     
    (255, 140, 0)  : 12, #      bag     
    (50, 205, 50)  : 9, #   neckwear     
    (220, 220, 220): 2, #    outer   
    (255, 0, 0)    : 13, #     hair   
    (127, 255, 212): 7, #     headwear  
    (0, 100, 0)    : 8, #   eyeglass    
    (255, 255, 0)  : 10, #       belt   
    (211, 211, 211): 5, #     pants    
    (144, 238, 144): 15, #    skin    
    (245, 222, 179): 11, #    footwear    
}

label2color = dict(zip(segm_table.values(), segm_table.keys()))

palette = []
for i in range(len(label2color)):
    palette.extend(list(label2color[i]))

def convert_segm(segm):
    segm = segm[:,:,:3]

    height, width = segm.shape[:2]
    new_segm = np.zeros((height, width), dtype=np.uint8)
    
    for color, label in segm_table.items():
        new_segm[np.prod(segm == color, axis=2)==True] = label
    
    segm_image = Image.fromarray(new_segm).convert('P')
    segm_image.putpalette(palette)        
        
    return segm_image
