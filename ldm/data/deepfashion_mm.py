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
                **kwargs):
        super().__init__(folder, **kwargs)
        self.root = Path(folder)
        images = glob(str(self.root/'images/*.jpg'))

        train, test = train_test_split(images, test_size=test_size, random_state=test_split_seed)
        self.images = train if is_train else test
        self.smpl_folder = self.root/'smpl'
        self.segm = glob(str(self.root/'segm/*.png'))
        self.texts = json.load(open(self.root/'captions.json'))
        self.image_sizes = image_sizes
        self.image_transform = T.Compose([
            #T.Pad((38,0), padding_mode='edge'),
            T.Resize(image_sizes),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.pad = pad
        self.pose = pose
        self.pad = None if self.pad is None else tuple(self.pad)

    def __getitem__(self, index):
        try:
            image_file = self.images[index]
            image_id = os.path.basename(image_file)

            # text
            text = self.texts[image_id]

            # image
            image = Image.open(image_file)
            if self.pad:
                image = T.Pad(self.pad, padding_mode='edge')(image)
            image = self.image_transform(image)
            data = {"image": image, "txt": text}

            # segmentation
            #segm_file = image_file.replace('images/','segm/').replace('.jpg', '_segm.png')
            #segm = np.array(Image.open(segm_file))

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
            print(e)
            #print(f"An exception occurred trying to load SMPL or Face embedding.")
            #print(f"Skipping index {ind}")
            return self.skip_sample(index)            
        
        
        return data
