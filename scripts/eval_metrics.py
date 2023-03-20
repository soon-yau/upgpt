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
from pytorch_msssim import ssim, ms_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
import subprocess


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
    )  
    
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--sample_dir",
        type=str,
        default="",
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
        gt_files = glob(os.path.join(self.gt_folder, '*.jpg')) + glob(os.path.join(self.gt_folder, '*.png'))
        sample_files = glob(os.path.join(self.sample_folder, '*.jpg')) + glob(os.path.join(self.sample_folder, '*.png'))

        if len(gt_files)!=len(sample_files):
            print(f'Missing result files {len(gt_files)-len(sample_files)}')
        self.data = sample_files
        self.image_transform = T.ToTensor()
    
    def __len__(self):
        return len(self.data)

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.sequential_sample(ind=ind)

    @abstractmethod
    def __getitem__(self, index):
        try:
            sample_file = self.data[index]
            gt_file = sample_file.replace(self.sample_folder, self.gt_folder)
            sample = self.image_transform(Image.open(sample_file))
            gt = self.image_transform(Image.open(gt_file))
            file_name = os.path.basename(sample_file)
        except Exception as e:            
            #print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)
        return gt, sample, file_name

def eval(gt_dir, sample_dir, batch_size, gpu=0):

    fnames = []
    ssim_scores = []
    #lpips_alex = []
    lpips_scores = []
    msssim_scores =[]
    
    device = f'cuda:{gpu}'
    #loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    ret = os.popen(f"python -m pytorch_fid {gt_dir} {sample_dir}  --device {device} --batch-size {batch_size}")
    loader = iter(DataLoader(Loader(gt_dir, sample_dir),
                        batch_size=batch_size, shuffle=False))

    for gt, sample, fname in tqdm(loader):
        fnames.extend(fname)
        sample = sample.to(device)
        gt = gt.to(device)
        ssim_scores.extend(ssim(sample, gt, data_range=1, size_average=False).cpu().numpy().tolist())
        msssim_scores.extend(ms_ssim(sample, gt, data_range=1, size_average=False).cpu().numpy().tolist())
        lpips_scores.extend(loss_fn_vgg(sample, gt).view(-1).detach().cpu().numpy().tolist())

    
    fid_str = ret.read()

    df = pd.DataFrame({'name':fnames, 
                        'SSIM':ssim_scores,
                        #'lpips_alex': lpips_alex,
                        'LPIPS': lpips_scores,
                        'MSSIM': msssim_scores
                        })
    log_dir = Path(sample_dir)/'../'
    df.to_csv(log_dir/'metrics.csv', index=False)


    text_file = open(str(log_dir/'metrics.txt'), "w")
    print('\n'+fid_str.split('\n')[0])
    text_file.write(fid_str)

    for metric in ['SSIM', 'MSSIM', 'LPIPS']:
        line = f"{metric}: {df[metric].mean()}"
        print(line)
        text_file.write(line+'\n')

    text_file.close()
        
def resize_image(root, size):
    
    image_files = glob(str(Path(root)/'**/*.jpg'), recursive=True)

    for image_file in image_files:
        image = Image.open(image_file)
        if image.size != tuple(size[::-1]):
            T.CenterCrop(size)(Image.open(image_file)).save(image_file)
        else:
            break

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    root = Path(args.dir) if args.dir else None
    gt_root = args.gt_dir if args.gt_dir else str(root/'gt')
    sample_root = args.sample_dir if args.sample_dir else str(root/'samples')
    
    #resize_image(sample_root, (512,256))
    #resize_image(gt_root, (512,256))

    eval(gt_root, sample_root, args.batch_size, args.gpu)
