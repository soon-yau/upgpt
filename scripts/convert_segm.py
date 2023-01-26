import sys
import os
from multiprocessing import Pool

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy
from utils.segment import convert_segm
from tqdm import tqdm

src_root = '/home/soon/datasets/deepfashion_inshop/img_highres/'
dst_root = '/home/soon/datasets/deepfashion_inshop/segm/'
mm_root = '/home/soon/datasets/deepfashion_multimodal/segm/' 

segm_files =  glob(os.path.join(src_root,'**/*_segment.png'),recursive=True)

def convert(segm_file):
    seg = convert_segm(np.array(Image.open(segm_file)))
    dst_file = segm_file.replace(src_root, dst_root).replace('_segment.png','_segm.png')
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    seg.save(dst_file)

with Pool(os.cpu_count()-1) as p:
    p.map(convert, segm_files)


mm_segm = glob(os.path.join(mm_root,'*_segm.png'))

count = 0
print('copy files from multimodal')
for m in tqdm(mm_segm):
    src_id = os.path.basename(m)
    src_id = src_id.replace('-','/')
    dst = os.path.join(dst_root, src_id)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        count+=1
        copy(m, dst)

total = len(glob(os.path.join(dst_root,'**/*_segm.png'),recursive=True))
print(f'#Files: Inshop={len(segm_files)}, Multimodal={count}, Total={total}')