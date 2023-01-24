import os, sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from abc import abstractmethod
import pickle
import numpy as np
from glob import glob
import argparse
import pdb
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
import subprocess


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
    )  
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/home/soon/datasets/deepfashion_inshop/pose_transfer_gt",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100
    )      
    return parser

class Loader(Dataset):
    def __init__(self, gt_folder, sample_folder, shuffle=False):
        super().__init__()
        self.gt_folder = str(Path(gt_folder).resolve())
        self.sample_folder = str(Path(sample_folder).resolve())
        gt_files = glob(os.path.join(self.gt_folder, '*.jpg'))
        sample_files = glob(os.path.join(self.sample_folder, '*.jpg'))

        if len(gt_files)==len(sample_files):
            print(f'Missing result files {len(gt_files)-len(sample_files)}')
        self.data = sample_files
        self.image_transform = T.ToTensor()
    
    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, index):
        sample_file = self.data[index]
        gt_file = sample_file.replace(self.sample_folder, self.gt_folder)
        sample = self.image_transform(Image.open(sample_file))
        gt = self.image_transform(Image.open(gt_file))
        file_name = os.path.basename(sample_file)
        return gt, sample, file_name


def eval(gt_dir, sample_dir, batch_size, gpu=0):

    fnames = []
    ssim_scores = []
    #lpips_alex = []
    lpips_vgg = []

    device = f'cuda:{gpu}'
    #loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    #ret = subprocess.Popen(['python','-m', 'pytorch_fid', gt_dir, sample_dir])
    ret = os.popen(f"python -m pytorch_fid {gt_dir} {sample_dir}")  
    fid_str = ret.read()

    loader = iter(DataLoader(Loader(gt_dir, sample_dir),
                        batch_size=batch_size, shuffle=False))
    for gt, sample, fname in tqdm(loader):
        fnames.extend(fname)
        sample = sample.to(device)
        gt = gt.to(device)
        ssim_scores.extend(ssim(sample, gt, data_range=1, size_average=False).cpu().numpy().tolist())
        lpips_vgg.extend(loss_fn_vgg(sample, gt).view(-1).detach().cpu().numpy().tolist())


    df = pd.DataFrame({'name':fnames, 
                        'ssim':ssim_scores,
                        #'lpips_alex': lpips_alex,
                        'lpips_vgg': lpips_vgg})
    log_dir = Path(sample_dir)/'../'
    df.to_csv(log_dir/'metrics.csv', index=False)


    text_file = open(str(log_dir/'metrics.txt'), "w")
    print('\n'+fid_str)    
    text_file.write(fid_str)

    for metric in ['ssim', 'lpips_vgg']:
        line = f"{metric}: {df[metric].mean()}"
        print(line)
        text_file.write(line+'\n')

    text_file.close()
        

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    eval(args.gt_dir, args.sample_dir, args.batch_size, args.gpu)