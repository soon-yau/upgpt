import os, sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from random import choice
from ldm.data.pose_utils import PoseVisualizer
from abc import abstractmethod
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from einops import rearrange
from glob import glob
import random
import json

class DeepfashionMMSegment:
    def __init__(self):
        
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
        self.segm_groups = [
            ['background'],
            [ 'hair', 'eyeglass','face','headwear'],
            [ 'top','outer','rompers'],
            ['skirt','dress','leggings','pants'],
            ['footwear','socks'],   
        ]

        self.segm_id_groups = []
        for g in self.segm_groups:
            self.segm_id_groups.append([self.label2id[l] for l in g])
                
        self.clip_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(size=224),
            T.CenterCrop(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
    def get_mask(self, image, segm, mask_ids):
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
        
        vertical = torch.mean(mask.to(torch.float32), dim=0)
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

        horizontal = torch.mean(mask.to(torch.float32), dim=1)

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

    def crop(self, image, # torch tensor
             mask, 
             margin=100):
        mask_range = self.get_mask_range(mask, margin)

        cropped = (image * mask)[:,mask_range['top']:mask_range['bottom'], 
                            mask_range['left']:mask_range['right']]
        
        return self.clip_transform(cropped)

    def forward(self, image, segm):

        image = T.ToTensor()(image)
        cropped_images = []
        for segm_group in self.segm_id_groups:
            mask = torch.from_numpy(self.get_mask(image, segm, segm_group))
            cropped = self.crop(image, mask)
            cropped_images.append(cropped)
            
        return torch.stack(cropped_images)

class Loader(Dataset):
    def __init__(self, folder, shuffle=False):
        super().__init__()
        self.shuffle = shuffle

    
    def __len__(self):
        return len(self.images)
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    @abstractmethod
    def __getitem__(self, ind):
        pass


class DeepFashionMM(Loader):
    
    def __init__(self, 
                folder, 
                image_sizes, 
                pose=None,
                is_train=True,
                test_size=64, 
                test_split_seed=None,
                pad=None,
                no_segm_percent=0.0, # percentage of samples without segmentation
                **kwargs):
        super().__init__(folder, **kwargs)
        self.root = Path(folder)
        images = glob(str(self.root/'images/*.jpg'))
        segm = glob(str(self.root/'segm/*.png'))
        
        # samples image without segm
        image_ids = set([os.path.basename(image).split('.jpg')[0] for image in images])
        segm_ids = set([os.path.basename(image).split('_segm.png')[0] for image in segm])
        no_segm_ids = image_ids ^ segm_ids
        np.random.seed(test_split_seed)
        select_no_segm_ids = np.random.choice(list(no_segm_ids), 
                                              size=int(no_segm_percent*len(segm_ids)), 
                                              replace=False)
        select_ids = list(select_no_segm_ids) + list(segm_ids)
        
        images = [f'{self.root}/images/{x}.jpg' for x in select_ids]
        
        train, test = train_test_split(images, test_size=test_size, random_state=test_split_seed)
        self.images = train if is_train else test
        self.smpl_folder = self.root/'smpl'
        self.segm = glob(str(self.root/'segm/*.png'))
        self.texts = json.load(open(self.root/'captions.json'))
        self.image_sizes = image_sizes
        self.image_transform = T.Compose([
            T.Resize(image_sizes),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.pad = pad
        self.pose = pose
        self.pad = None if self.pad is None else tuple(self.pad)
        self.segmentor = DeepfashionMMSegment()
        
    def __getitem__(self, index):
        try:
            data = {}
            
            image_file = self.images[index]
            image_id = os.path.basename(image_file)

            # text
            text = self.texts[image_id]

            # image
            image = Image.open(image_file)
            
            # segmentation
            segm_file = image_file.replace('images/','segm/').replace('.jpg', '_segm.png')
            
            segm = np.array(Image.open(segm_file))

            styles = self.segmentor.forward(image, segm)
            data.update({'styles':styles})
        
            # image
            if self.pad:
                image = T.Pad(self.pad, padding_mode='edge')(image)
            image = self.image_transform(image)
            data.update({"image": image, "txt": text})

            # SMPL
            if self.pose == 'smpl':
                smpl_image_file = image_file.replace('/images/','/smpl/')
                smpl_file = smpl_image_file.replace('.jpg','.p')

                smpl_image = Image.open(smpl_image_file)
                smpl_image = self.image_transform(smpl_image)

                with open(smpl_file, 'rb') as f:
                    smpl_params = pickle.load(f)
                    pred_pose = smpl_params[0]['pred_body_pose']
                    pred_betas = smpl_params[0]['pred_betas']
                    pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
                    smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
                    smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))

                data.update({'smpl':smpl_pose, 'smpl_image':smpl_image})


            return data

        except Exception as e:
            #print(e)
            #print(f"An exception occurred trying to load SMPL or Face embedding.")
            #print(f"Skipping index {ind}")
            return self.skip_sample(index)            
        
        
        return data