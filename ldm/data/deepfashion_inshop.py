import os, sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from random import choice
from ldm.data.segm_utils import LIPSegmenter
from abc import abstractmethod
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from einops import rearrange
from glob import glob
import random
import json

#from scripts.segment import segm_groups

style_names = ['face', 'hair', 'headwear', 'background', 'top', 'outer', 'bottom', 'shoes', 'accesories']

class Loader(Dataset):
    def __init__(self, folder, shuffle=False):
        super().__init__()
        self.shuffle = shuffle
    
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


class DeepFashion(Loader):
    
    def __init__(self, 
                folder,
                image_dir,
                data_file, 
                image_sizes=None, 
                pose=None,
                is_train=True,
                test_size=0, 
                test_split_seed=None,
                pad=None,
                **kwargs):
        super().__init__(folder, **kwargs)
        self.root = Path(folder)
        self.image_root = self.root/image_dir
        self.pose_root = self.root/'smpl_256'
        self.style_root = self.root/'styles'
        self.segm_root = self.root/'lip_segm_256'
        self.texts = json.load(open(self.root/'captions.json'))
        self.segmenter = LIPSegmenter()

        self.df = pd.read_csv(data_file)
        # temporary drop those without poses
        self.df = self.df.drop(self.df[self.df.pose.isnull()].index)
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.drop(self.df[self.df.styles.isnull()].index)
        self.df = self.df.reset_index(drop=True)
        if test_size != 0:
            train, test = train_test_split(self.df, test_size=test_size, random_state=test_split_seed)
            self.df = train if is_train else test

        self.image_sizes = image_sizes
        transform_list = [T.Resize(image_sizes)] if image_sizes else []
        self.image_transform = T.Compose(transform_list + [
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        self.pose = pose
        self.pad = None if pad is None else tuple(pad)
        
        self.style_names = style_names #['face', 'background', 'top', 'bottom', 'shoes', 'accesories'] 
        self.clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = T.Compose([
            T.ToTensor(),
            self.clip_norm
        ])

        self.mask_transform = T.Compose([
            T.Resize(size=(32,24), interpolation=T.InterpolationMode.NEAREST),    
            T.ToTensor(),
            T.Lambda(lambda x: x * 2. - 1.)
        ])

        self.loss_w_transform = T.Compose([
            T.Resize(size=(32,24), interpolation=T.InterpolationMode.NEAREST),    
            T.ToTensor(),
        ])        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            row = self.df.iloc[index]
            '''
            for k in ['text', 'image', 'pose', 'styles']:
                if type(row[k]) != str:
                    print(index, k, row[k])
            '''
            # text
            text = self.texts[row['text']]
            
            # segmentation map
            segm_file = str(self.segm_root/row['image']).replace('.jpg','.png')
            segm = np.array(Image.open(segm_file))
            loss_weight = self.segmenter.get_mask(segm, 
                                    {'Background':0.5, 'Left-arm':2.0, 'Right-arm':2.0, 'Face':5.0})
            loss_weight = self.loss_w_transform(Image.fromarray(loss_weight))

            # image
            image = Image.open(self.image_root/row['image'])            
            image = self.image_transform(image)

            # style images
            style_images = []
            for style_name in self.style_names:
                f_path = self.style_root/row['styles']/f'{style_name}.jpg'
                if f_path.exists():
                    style_image = self.clip_transform((Image.open(f_path)))
                else:
                    style_image = self.clip_norm(torch.zeros(3, 224, 224))
                style_images.append(style_image)
            style_images = torch.stack(style_images)  

            data = {"image": image, "txt": text, "styles":style_images}


            # image
            if self.pad:
                image = T.Pad(self.pad, padding_mode='edge')(image)

            # SMPL
            if self.pose == 'smpl':
                pose_path = str(self.pose_root/row['pose'])

                smpl_image_file = pose_path + '.jpg'
                smpl_file = pose_path + '.p'
                smpl_image = Image.open(smpl_image_file)
                smpl_image = self.image_transform(smpl_image)
                
                mask_file = pose_path + '_mask.png'
                mask_image = Image.open(mask_file)
                person_mask = self.mask_transform(mask_image)
                with open(smpl_file, 'rb') as f:
                    smpl_params = pickle.load(f)
                    pred_pose = smpl_params[0]['pred_body_pose']
                    pred_betas = smpl_params[0]['pred_betas']
                    pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
                    #pred_camera = T.ToTensor()(pred_camera).view((1,-1))
                    # camera has 3 parameters (z, x, y) where 0 is center
                    # z: +ve is zoom in. For x and y, +ve is right and up respectively
                    smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
                    smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))

                data.update({'smpl':smpl_pose, 'smpl_image':smpl_image, 
                            'person_mask':person_mask, 'loss_w':loss_weight})



        except Exception as e:
            #print(f"Skipping index {index}")
            return self.skip_sample(index)            

        
        return data

class DeepFashionTest(Loader):
    
    def __init__(self, 
                folder,
                image_dir,
                data_file,
                test_file,                 
                pose=None,
                max_size=None,
                image_sizes=None, 
                pad=None,
                **kwargs):
        super().__init__(folder, **kwargs)
        self.root = Path(folder)
        self.image_root = self.root/image_dir
        self.pose_root = self.root/'smpl_256'
        self.style_root = self.root/'styles'
        self.texts = json.load(open(self.root/'captions.json'))

        self.df = pd.read_csv(data_file)
        # temporary drop those without poses
        self.df = self.df.drop(self.df[self.df.pose.isnull()].index)
        self.df = self.df.drop(self.df[self.df.styles.isnull()].index)
        self.df = self.df.drop(self.df[self.df['image']==''].index)
        self.df = self.df.reset_index(drop=True)

        self.df = self.df.set_index('image')
        # test config fiile
        self.test_df = pd.read_csv(test_file)
        if max_size:
            self.test_df = self.test_df.iloc[:max_size]

        self.image_sizes = image_sizes
        transform_list = [T.Resize(image_sizes)] if image_sizes else []
        self.image_transform = T.Compose(transform_list + [
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        self.pose = pose
        self.pad = None if pad is None else tuple(pad)
        
        self.style_names = style_names#['face', 'background', 'top', 'bottom', 'shoes', 'accesories'] #list(segm_groups.keys())
        self.clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = T.Compose([
            T.ToTensor(),
            self.clip_norm
        ])

        self.mask_transform = T.Compose([
            T.Resize(size=(32,24), interpolation=T.InterpolationMode.NEAREST),    
            T.ToTensor(),
            T.Lambda(lambda x: x * 2. - 1.)
        ])

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, index):
        try:
            row = self.test_df.iloc[index]
            
            # src
            source = self.df.loc[row['from']]
            src_path = str(self.image_root/source.name)
            source_image = self.image_transform(Image.open(src_path))
            styles_path = source['styles']

            # target
            target = self.df.loc[row['to']]
            target_path = str(self.image_root/target.name)
            text = self.texts[target.text]
            pose_path = target.pose
            target_image = self.image_transform(Image.open(target_path))

            # style images
            style_images = []
            for style_name in self.style_names:
                f_path = self.style_root/styles_path/f'{style_name}.jpg'
                
                if f_path.exists():
                    style_image = self.clip_transform((Image.open(f_path)))
                else:
                    style_image = self.clip_norm(torch.zeros(3, 224, 224))
                style_images.append(style_image)
            style_images = torch.stack(style_images)  

            data = {"test_id":  index,
                    "src_image": source_image,
                    "image": target_image, 
                    "txt": text, 
                    "styles":style_images}

            # image
            if self.pad:
                image = T.Pad(self.pad, padding_mode='edge')(image)
  
            # SMPL
            if self.pose == 'smpl':
                pose_path = str(self.pose_root/pose_path)
                smpl_image_file = pose_path + '.jpg'
                smpl_file = pose_path + '.p'
                smpl_image = Image.open(smpl_image_file)
                smpl_image = self.image_transform(smpl_image)

                mask_file = pose_path + '_mask.png'
                mask_image = Image.open(mask_file)
                person_mask = self.mask_transform(mask_image)
                
                with open(smpl_file, 'rb') as f:
                    smpl_params = pickle.load(f)
                    pred_pose = smpl_params[0]['pred_body_pose']
                    pred_betas = smpl_params[0]['pred_betas']
                    pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
                    #pred_camera = T.ToTensor()(pred_camera).view((1,-1))
                    smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
                    smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))


                data.update({'smpl':smpl_pose, 'smpl_image':smpl_image, 
                            'person_mask':person_mask})


            return data

        except Exception as e:
            print(f"Skipping index {index}", e)
            return self.skip_sample(index)            
        
        
        return data

class DeepFashionImageOnly(Loader):
    
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
        images = glob(str(self.root/'**/*.jpg'), recursive=True)
        
        train, test = train_test_split(images, test_size=test_size, random_state=test_split_seed)
        self.images = train if is_train else test
        self.image_sizes = image_sizes
        self.image_transform = T.Compose([
            T.Resize(image_sizes, antialias=True),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.pad = pad
        self.pose = pose
        self.pad = None if self.pad is None else tuple(self.pad)
        
    def __getitem__(self, index):
        try:

            image_file = self.images[index]
            image_id = os.path.basename(image_file)
            image = Image.open(image_file)
            # image
            if self.pad:
                image = T.Pad(self.pad, padding_mode='edge')(image)
            image = self.image_transform(image)
            data = {"image": image}

        except Exception as e:
            #print(e)
            #print(f"An exception occurred trying to load SMPL or Face embedding.")
            #print(f"Skipping index {ind}")
            return self.skip_sample(index)            
        
        
        return data