import os, sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from random import choice
from ldm.data.segm_utils import DeepfashionMMSegmenter
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

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_name(src, dst):
    src = convert_fname(src)
    dst = convert_fname(dst)
    return src + '___' + dst

def list_subdirectories(path):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(path):
        if not dirnames:
            subdirectories.append(dirpath)
    return subdirectories
        

class DeepFashionPair(Loader):
    
    def __init__(self, 
                folder,
                image_dir,
                pair_file, # from, to 
                data_file, # point to style features and text
                df_filter=None,
                image_size=[256, 192], 
                f=8,
                resize_size=None,
                pad=None,
                max_size=0, 
                test_split_seed=None,
                input_mask_type='mask',
                loss_weight=None,
                image_only=False,
                dropout=None,
                random_style=False,
                men_factor=None,
                **kwargs):
        super().__init__(folder, **kwargs)
        assert input_mask_type in ['mask', 'smpl', 'bbox']
        self.image_only = image_only
        self.input_mask_type = input_mask_type
        self.root = Path(folder)
        self.image_root = self.root/image_dir
        self.pose_root = self.root/'smpl_256' if self.input_mask_type in ['mask','bbox'] else self.root/'smpl'
        self.style_root = self.root/'styles'
        self.segm_root = self.root/'segm_256'
        self.texts = json.load(open(self.root/'captions.json'))
        self.map_df = pd.read_csv(data_file)
        self.map_df.set_index('image', inplace=True)
        self.vae_z_size = tuple([x//f for x in image_size])
        self.loss_weight = loss_weight
        self.dropout = dropout
        self.random_style = random_style
        dfs = [pd.read_csv(f) for f in pair_file]
        self.df = pd.concat(dfs, ignore_index=True)
        if df_filter:
            self.df = self.df[self.df[df_filter]==True].reset_index()
        
        if max_size != 0:
            _, self.df = train_test_split(self.df, test_size=max_size, random_state=test_split_seed)
        
        if men_factor:
            self.df['gender'] = self.df['from'].map(lambda x: x.split('/')[0])
            men_df = pd.concat([self.df[self.df['gender']=='MEN']]*men_factor)
            self.df = pd.concat([self.df, men_df]).reset_index()
                    
        ''' pad and resize '''
        pre_transform = []
        if resize_size:
            pre_transform.append(T.Resize(resize_size))
        if pad:
            pre_transform.append(T.Pad(tuple(pad)))
        self.resize_pad_image = T.Compose(pre_transform)

        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

        ''' '''

        self.clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))
        self.clip_transform = T.Compose([
            T.ToTensor(),
            self.clip_norm
        ])


        self.loss_w_transform = T.Compose([
            T.Resize(size=self.vae_z_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])    
        
        if self.input_mask_type in ['mask', 'bbox'] :
            self.mask_transform = T.Compose([
                T.Resize(size=self.vae_z_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
                T.Lambda(lambda x: x * 2. - 1.)
            ])                
        else :
            self.mask_transform = T.Compose([
                T.Resize(size=self.vae_z_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Lambda(lambda x: torch.mean(x,0, keepdim=True)  * 2. - 1.)
            ])             

        self.smpl_image_transform = T.Compose([
            #T.Resize(size=256),
            T.CenterCrop(size=(256, 192))])
        
        self.segmenter = DeepfashionMMSegmenter()
        self.style_names = style_names

    def __len__(self):
        return len(self.df)

    def get_bbox(self, mask):
        x = np.nonzero(np.mean(mask,1))[0]
        xmin, xmax = x[0], x[-1]
        y = np.nonzero(np.mean(mask,0))[0]
        ymin, ymax = y[0], y[-1]
        bbox = np.zeros_like(mask, np.uint8)
        bbox[xmin:xmax+1, ymin:ymax+1] = 1
        return bbox
    
    def __getitem__(self, index):
        try:
            
            row = self.df.iloc[index]
            data = {}
            # target - get text, person_mask, pose, 
            target = self.map_df.loc[row['to']]           

            target_path = str(self.image_root/target.name)
            target_image = self.image_transform(self.resize_pad_image(Image.open(target_path)))
            text = self.texts.get(target.text, '')
            
            data.update({"image": target_image, "txt": text})
            if self.image_only:
                return data
            fname = get_name(row['from'], row['to'])
            # source - get fashion styles
            source = self.map_df.loc[row['from']]
            src_path = str(self.image_root/source.name)
            source_image = Image.open(src_path)
            #styles_dict = self.segmenter.forward(source_image, segm)
            #styles = torch.stack(list(styles_dict.values()))            
            styles_path = source['styles']
            if styles_path == np.nan:
                return self.skip_sample(index)
            
            drop_style = False
            if self.dropout:
                if random.uniform(0,1) < self.dropout:
                    drop_style = True

            full_styles_path = self.style_root/source['styles']
            if self.random_style:
                full_styles_path = Path(random.choice(list_subdirectories(full_styles_path.parent.parent.parent)))

            style_images = []
            for style_name in self.style_names:
                f_path = full_styles_path/f'{style_name}.jpg'
                if f_path.exists() and not drop_style:
                    style_image = self.clip_transform((Image.open(f_path)))
                else:
                    style_image = self.clip_norm(torch.zeros(3, 224, 224))
                style_images.append(style_image)
            style_images = torch.stack(style_images)  

            data.update({"fname": fname, 
                    "src_image": self.image_transform(self.resize_pad_image(source_image)),
                    "styles": style_images})


            # SMPL            
            pose_path = str(self.pose_root/target.pose)
            smpl_image_file = pose_path + '.jpg'
            smpl_file = pose_path + '.p'
            smpl_image = self.smpl_image_transform(Image.open(smpl_image_file))
            if self.input_mask_type=='mask':
                mask_file = pose_path + '_mask.png'
                mask_image = Image.open(mask_file)
                person_mask = self.mask_transform(mask_image)
            elif self.input_mask_type=='bbox':
                mask_file = pose_path + '_mask.png'
                mask_image = self.get_bbox(np.array(Image.open(mask_file)))
                '''
                Should multiply by 255 but keep the bug to be backward compatible with trained model.
                person_mask = self.mask_transform(Image.fromarray(mask_image)*255)
                '''                
                person_mask = self.mask_transform(Image.fromarray(mask_image))
            else:
                person_mask = self.mask_transform(smpl_image)
          
            smpl_image = self.image_transform(smpl_image)

            with open(smpl_file, 'rb') as f:
                smpl_params = pickle.load(f)
                pred_pose = smpl_params[0]['pred_body_pose']
                pred_betas = smpl_params[0]['pred_betas']
                pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
                smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
                smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))

            data.update({'smpl':smpl_pose, 
                         'smpl_image':smpl_image, 
                         'person_mask':person_mask
                         })

            if self.loss_weight:
                segm_path = str(self.segm_root/target.name).replace('.jpg','_segm.png')            
                segm = np.array(Image.open(segm_path))

                loss_weight = self.segmenter.get_mask(segm, self.loss_weight)
                loss_weight = self.loss_w_transform(Image.fromarray(loss_weight))

                data.update({'loss_w':loss_weight})

            return data

        except Exception as e:            
            #print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)


class DeepFashionSample(DeepFashionPair):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        # source - get fashion styles
        source = self.map_df.loc[index]
        src_path = str(self.image_root/source.name)
        source_image = Image.open(src_path)

        #styles_dict = self.segmenter.forward(source_image, segm)
        #styles = torch.stack(list(styles_dict.values()))            
        full_styles_path = self.style_root/source['styles']
        if self.random_style:
            full_styles_path = Path(random.choice(list_subdirectories(full_styles_path.parent.parent.parent)))

        style_images = []
        for style_name in self.style_names:
            f_path = full_styles_path/f'{style_name}.jpg'

            if f_path.exists():
                style_image = self.clip_transform((Image.open(f_path)))
            else:
                style_image = self.clip_norm(torch.zeros(3, 224, 224))
            style_images.append(style_image)
        style_images = torch.stack(style_images)  


        data = {#"fname": fname, 
                "src_image": self.image_transform(self.resize_pad_image(source_image)),
                "styles": style_images}

        # target - get text, person_mask, pose, 
        target = self.map_df.loc[index]

        text = self.texts.get(target.text, '')

        target_path = str(self.image_root/target.name)
        target_image = self.image_transform(self.resize_pad_image(Image.open(target_path)))

        data.update({"image": target_image, "txt": text,})

        # image
        #if self.pad:
        #    image = T.Pad(self.pad, padding_mode='edge')(image)

        # SMPL            
        pose_path = str(self.pose_root/target.pose)
        smpl_image_file = pose_path + '.jpg'
        smpl_file = pose_path + '.p'
        smpl_image = self.smpl_image_transform(Image.open(smpl_image_file))
        if self.input_mask_type=='mask':
            mask_file = pose_path + '_mask.png'
            mask_image = Image.open(mask_file)
            person_mask = self.mask_transform(mask_image)
        elif self.input_mask_type=='bbox':
            mask_file = pose_path + '_mask.png'
            mask_image = self.get_bbox(np.array(Image.open(mask_file)))
            '''
            Should multiply by 255 but keep the bug to be backward compatible with trained model.
            person_mask = self.mask_transform(Image.fromarray(mask_image)*255)
            '''
            person_mask = self.mask_transform(Image.fromarray(mask_image))
        else:
            person_mask = self.mask_transform(smpl_image)

        smpl_image = self.image_transform(smpl_image)

        with open(smpl_file, 'rb') as f:
            smpl_params = pickle.load(f)
            pred_pose = smpl_params[0]['pred_body_pose']
            pred_betas = smpl_params[0]['pred_betas']
            pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
            smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
            smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))


        data.update({'smpl':smpl_pose, 
                     'smpl_image':smpl_image, 
                     'person_mask':person_mask})


        return data


class DeepFashionSuperRes(DeepFashionPair):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lr_root = self.root/'recon_256'
        self.style_names = style_names
        self.lr_transform = T.Compose([
            T.Resize(size=self.vae_z_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2. - 1.,)])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        try:            
            row = self.df.iloc[index]
            data = {}
            # source - get fashion styles
            source = self.map_df.loc[row['from']]
            src_path = str(self.image_root/source.name)
            source_image = Image.open(src_path)
            lr_image =  Image.open(str(self.lr_root/source.name))
            text = self.texts.get(source.text, '')
            
            full_styles_path = self.style_root/source['styles']

            style_images = []
            for style_name in self.style_names:
                f_path = full_styles_path/f'{style_name}.jpg'

                if f_path.exists():
                    style_image = self.clip_transform((Image.open(f_path)))
                else:
                    style_image = self.clip_norm(torch.zeros(3, 224, 224))
                style_images.append(style_image)
            style_images = torch.stack(style_images)  

            lr = self.lr_transform(lr_image)
            lr_image = rearrange(lr, 'c h w -> h w c')
            data = {"lr": lr,
                    "lr_image": lr_image,
                    "image": self.image_transform(self.resize_pad_image(source_image)),
                    "styles": style_images,
                    "txt": text}
            return data

        except Exception as e:            
            #print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)
        

class DeepFashionSuperResSampling(DeepFashionPair):
    
    def __init__(self, **kwargs):
        lr_dir = kwargs.pop('lr_dir')
        super().__init__(**kwargs)
        
        self.lr_root = Path(lr_dir)
        self.style_names = style_names
        self.lr_transform = T.Compose([
            T.Pad((8,0), padding_mode='edge'),
            T.Resize(size=self.vae_z_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2. - 1.,)])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        ext = '.jpg'
        try:            
            row = self.df.iloc[index]
            data = {}
            # source - get fashion styles
            source = self.map_df.loc[row['from']]
            src_path = str(self.image_root/source.name)
            source_image = Image.open(src_path)

            fname = get_name(row['from'], row['to'])
            #print(str(self.lr_root/fname)+ext)
            lr_image =  Image.open(str(self.lr_root/fname)+ext)
            text = self.texts.get(source.text, '')
            
            full_styles_path = self.style_root/source['styles']

            style_images = []
            for style_name in self.style_names:
                f_path = full_styles_path/f'{style_name}.jpg'

                if f_path.exists():
                    style_image = self.clip_transform((Image.open(f_path)))
                else:
                    style_image = self.clip_norm(torch.zeros(3, 224, 224))
                style_images.append(style_image)
            style_images = torch.stack(style_images)  

            lr = self.lr_transform(lr_image)
            lr_image = rearrange(lr, 'c h w -> h w c')
            data = {"fname": fname,
                    "lr": lr,
                    "lr_image": lr_image,
                    "image": self.image_transform(self.resize_pad_image(source_image)),
                    "styles": style_images,
                    "txt": text}
            #print(index, row['from'], row['to'])
            return data

        except Exception as e:            
            #print(f"Skipping index {index}", e)
            # sys.exit()
            return self.skip_sample(index)
        