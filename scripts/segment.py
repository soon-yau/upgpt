import os, sys
from glob import glob
import numpy as np

from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map 

from scripts.segm_utils import DeepfashionMMSegment


if __name__ == "__main__":
    segmentor = DeepfashionMMSegmenter(device='cuda:0')

    image_root = '/home/soon/datasets/deepfashion_inshop/img_highres/'
    segm_root = '/home/soon/datasets/deepfashion_inshop/segm/'
    dst_root = '/home/soon/datasets/deepfashion_inshop/styles/'
    segm_files = glob(os.path.join(segm_root,'**/*_segm.png'),recursive=True)

    #for segm_file in tqdm(segm_files[:]):
    def extract(segm_file):
        image_file = segm_file.replace('_segm.png','.jpg').replace(segm_root, image_root)
        image = np.array(Image.open(image_file))
        segm = np.array(Image.open(segm_file))
        cropped = segmentor.forward(image, segm)
        file_id = segm_file.replace('_segm.png','')
        path, fname = os.path.split(file_id)
        dst_dir = os.path.join(path, fname.replace('_','/',1)).replace(segm_root, dst_root)
        os.makedirs(dst_dir, exist_ok=True)
        for k, v in cropped.items():
            if v!= None:
                crop_file = os.path.join(dst_dir, k+'.jpg')
                v.save(crop_file)

    #with Pool(os.cpu_count()-1) as p:
    with Pool(8) as p:
        p.map(extract, segm_files)
    #process_map(extract, segm_files, max_workers=8)
    print(f'Processed {len(segm_files)} files.')