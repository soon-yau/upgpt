import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
import PIL
import pandas as pd
from random import choice
from ldm.data.pose_utils import PoseVisualizer
from abc import abstractmethod
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from einops import rearrange

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]

class Loader(Dataset):
    def __init__(self, pickle_file, folder, shuffle):
        super().__init__()
        self.shuffle = shuffle
        self.df = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.df)

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

'''
class DeepFashionKeypoint(Loader):

    def __init__(self, pickle_file, folder, is_train, shuffle=False):
        super().__init__(pickle_file, folder,shuffle)

        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 
        if not is_train:
            _, self.df = train_test_split(self.df, test_size=4)        
        self.root_dir = Path(folder)
        self.pose_visualizer = PoseVisualizer('keypoint', (256,256))
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = self.root_dir / sample.image
        descriptions = sample.text.copy()
        keypoints = sample.keypoints.copy()
        keypoints = T.ToTensor()(keypoints).view((1,-1))
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load captions.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.image_transform(image)
        pose = self.pose_visualizer.convert(sample.keypoints)*2.0-1.0
        return image, description, pose, keypoints
'''

class DeepFashionSMPL(Loader):

    def __init__(self, pickle_file, folder, smpl_folder, is_train, shuffle=False, random_drop=0.0):
        super().__init__(pickle_file, folder, shuffle)
        self.random_drop = random_drop  # drop smpl condition 
        self.smpl_folder = Path(smpl_folder)
        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 

        if not is_train:
            _, self.df = train_test_split(self.df, test_size=10)
        self.root_dir = Path(folder)
        self.image_transform = T.Compose([
            #T.Resize((512,512)),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
            #T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = self.root_dir / sample.image
        smpl_image_file = str(self.smpl_folder/sample.image)
        smpl_file = smpl_image_file.replace('.jpg','.p')
        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.image_transform(image)

        try:
            smpl_image = PIL.Image.open(smpl_image_file)
            #smpl_image = smpl_image.convert('RGB') if smpl_image.mode != 'RGB' else smpl_image
            smpl_image = self.image_transform(smpl_image)

            with open(smpl_file, 'rb') as f:
                smpl_params = pickle.load(f)
                pred_pose = smpl_params[0]['pred_body_pose']
                pred_betas = smpl_params[0]['pred_betas']
                smpl_pose = np.concatenate((pred_pose, pred_betas), axis=1)
                smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))
        except Exception as e:
            print(e)
            print(f"An exception occurred trying to load SMPL.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        descriptions = sample.text.copy()
        keypoints = sample.keypoints.copy()
        keypoints = T.ToTensor()(keypoints).view((1,-1))
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load captions.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        if self.random_drop > 0 and np.random.uniform() < self.random_drop:
            smpl_image = torch.zeros_like(smpl_image)
            smpl_pose = torch.zeros_like(smpl_pose)
        return {"image": image, "txt": description, 'smpl':smpl_pose, 'smpl_image':smpl_image}
        #return image, description, smpl_image, smpl_pose


class DeepFashionKeypoint(Loader):

    def __init__(self, pickle_file, folder, is_train, shuffle=False):
        super().__init__(pickle_file, folder,shuffle)

        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 
        if not is_train:
            _, self.df = train_test_split(self.df, test_size=10)
        self.root_dir = Path(folder)
        self.pose_visualizer = PoseVisualizer('keypoint', (256,256))
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = self.root_dir / sample.image
        descriptions = sample.text.copy()
        keypoints = sample.keypoints.copy()
        keypoints = T.ToTensor()(keypoints).view((1,-1))
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load captions.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.image_transform(image)
        pose = self.pose_visualizer.convert(sample.keypoints)*2.0-1.0
        return {"image": image, "txt": description, 'pose':keypoints, 'pose_image':pose}

'''
class DeepFashionSMPL_SR(Loader):

    def __init__(self, pickle_file, folder, smpl_folder, is_train, shuffle=False, lowres=64):
        super().__init__(pickle_file, folder, shuffle)
        self.lowres = tuple((lowres, lowres))
        self.smpl_folder = Path(smpl_folder)
        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 

        if not is_train:
            _, self.df = train_test_split(self.df, test_size=4)
        self.root_dir = Path(folder)
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = self.root_dir / sample.image
        smpl_image_file = str(self.smpl_folder/sample.image)
        smpl_file = smpl_image_file.replace('.jpg','.p')

        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        lowres_image = image.resize(self.lowres).resize(image.size)
        lowres_image = self.image_transform(lowres_image)
        image = self.image_transform(image)

        try:
            with open(smpl_file, 'rb') as f:
                smpl_params = pickle.load(f)
                pred_pose = smpl_params[0]['pred_body_pose']
                pred_betas = smpl_params[0]['pred_betas']
                smpl_pose = np.concatenate((pred_pose, pred_betas), axis=1)
                smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))
        except Exception as e:
            print(e)
            print(f"An exception occurred trying to load SMPL.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        descriptions = sample.text.copy()
        keypoints = sample.keypoints.copy()
        keypoints = T.ToTensor()(keypoints).view((1,-1))
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load captions.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        return image, description, lowres_image, smpl_pose
'''