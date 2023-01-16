import os, sys
from pathlib import Path
from PIL import Image
import pandas as pd
from random import choice
import numpy as np
from glob import glob
from torchvision import transforms as T
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

root = Path('./datasets/deepfashion_inshop/')
highres_dir = root/'img_highres'
highres_images = glob(str(highres_dir/'**/*.jpg'), recursive=True)

def resize(highres_path):
    pad = (38, 0)
    image_highres = Image.open(highres_path)
    image_highres = T.Pad(pad, padding_mode='edge')(image_highres)
    image_512 = image_highres.resize((384, 512), Image.Resampling.LANCZOS)
    image_256 = image_512.resize((192, 256), Image.Resampling.LANCZOS)
    
    path_512 = highres_path.replace('img_highres', 'img_512')
    path_256 = highres_path.replace('img_highres', 'img_256')
    os.makedirs(os.path.split(path_512)[0], exist_ok=True)
    os.makedirs(os.path.split(path_256)[0], exist_ok=True)
    image_512.save(path_512)
    image_256.save(path_256)

process_map(resize, highres_images, max_workers=8);