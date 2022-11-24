import os
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

class DeepFashionSMPL(Loader):

    def __init__(self, pickle_file, folder, smpl_folder, is_train, shuffle=False, \
                random_drop=0.0, test_size=0.005, test_split_random=8):
        super().__init__(pickle_file, folder, shuffle)
        self.random_drop = random_drop  # drop smpl condition 
        self.smpl_folder = Path(smpl_folder)
        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 

        train, test = train_test_split(self.df, test_size=test_size, random_state=test_split_random)
        self.df = train if is_train else test

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
                pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
                smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
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

    def __init__(self, pickle_file, folder, is_train, shuffle=False, test_size=0.005, test_split_random=8):
        super().__init__(pickle_file, folder,shuffle)

        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 
        train, test = train_test_split(self.df, test_size=test_size, random_state=test_split_random)
        self.df = train if is_train else test
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
        pose_image = self.pose_visualizer.convert(sample.keypoints)
        pose_image = rearrange(pose_image * 2. - 1., 'c h w -> h w c')

        return {"image": image, "txt": description, 'pose':keypoints, 'pose_image':pose_image}

class DeepFashionKeypointFaceEmbed(Loader):

    def __init__(self, pickle_file, folder, is_train, shuffle=False, 
                random_drop = 0.0, test_size=0.005, test_split_random=8):
        super().__init__(pickle_file, folder,shuffle)
        self.random_drop = random_drop
        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1] 
        train, test = train_test_split(self.df, test_size=test_size, random_state=test_split_random)
        self.df = train if is_train else test
        self.root_dir = Path(folder)
        self.pose_visualizer = PoseVisualizer('keypoint', (256,256))
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = str(self.root_dir / sample.image)
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

        image = PIL.Image.open(image_file)
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.image_transform(image)
        pose_image = self.pose_visualizer.convert(sample.keypoints)
        pose_image = rearrange(pose_image * 2. - 1., 'c h w -> h w c')

        # random drop pose
        if self.random_drop > 0 and np.random.uniform() < self.random_drop:
            pose_image = torch.zeros_like(pose_image)
            keypoints = torch.zeros_like(keypoints)

        # Load embedding
        try:
            face_file = image_file.replace('img_256', 'face')
            face_image = PIL.Image.open(face_file)
            face_image = T.Resize((64,64))(face_image)
            face_image = self.image_transform(face_image)
            embed_file = face_file.replace('.jpg', '.p')
            with open(embed_file, 'rb') as f:
                embed = pickle.load(f).astype(np.float32)
                embed = T.ToTensor()(np.expand_dims(embed, 0)).view((1,-1))
        except Exception as e:
            return self.skip_sample(ind)

        if self.random_drop > 0 and np.random.uniform() < self.random_drop:
            face_image = torch.zeros_like(face_image)
            embed = torch.zeros_like(embed)

        return {"image": image, "txt": description, 'pose':keypoints, 'pose_image':pose_image, \
                "face_image":face_image, 'face_embed':embed}

class DeepFashionImages:
    def __init__(self, pickle_files, folders, is_train, test_size=48, test_split_random=8):
        super().__init__()
        self.shuffle = is_train 
        dfs = []
        for pickle_file, folder in zip(pickle_files, folders):
            df = pd.read_pickle(pickle_file)
            df.image = df.image.map(lambda x: os.path.join(folder,x))
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        self.df['num_keypoints']=self.df.keypoints.map(lambda x: x.shape[0])
        self.df = self.df[self.df['num_keypoints']==1]
        train, test = train_test_split(self.df, test_size=test_size, random_state=test_split_random)
        self.df = train if is_train else test
        self.df.reset_index(drop=True, inplace=True)

        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

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

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = sample.image
        image = PIL.Image.open(str(image_file))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.image_transform(image)
        return {"image": image}

