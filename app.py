import streamlit as st
st.set_page_config(layout="wide")

import os
import pickle
from PIL import Image
from pathlib import Path
from shutil import copy
import pandas as pd
import numpy as np
from glob import glob

import torch
from torchvision import transforms as T
from einops import rearrange
from omegaconf import OmegaConf
from ldm.data.generate_utils import InferenceModel, draw_styles, convert_fname, interp_mask

data_root = Path('/home/soon/datasets/deepfashion_inshop')
image_root = data_root/'img_512'
styles_root = data_root/'styles'
cache_root = Path('app_cache')
local_style_root = cache_root/'styles'
local_pose_root = cache_root/'pose'
local_lowres_root = cache_root/'samples_lowres'

os.makedirs(local_style_root, exist_ok=True)
os.makedirs(local_lowres_root, exist_ok=True)
pose_folders = sorted([x[0] for x in os.walk(local_pose_root)][1:])
pose_images = [Image.open(Path(x)/'pose.jpg') for x in pose_folders]

def clear_image_cache():
    for x in glob(str(local_lowres_root/'*')):
        os.remove(x)

def get_image_number():
    image_files = [os.path.split(x)[1] for x in glob(str(local_lowres_root/'*.jpg'))]
    if len(image_files) == 0:
        return 0
    file_id = max([int(f.split('_')[1].split('.jpg')[0]) for f in image_files])
    return file_id + 1

def get_samples():
    return [Image.open(x) for x in sorted(glob(str(local_lowres_root/'*.jpg')))]

map_df = pd.read_csv("data/deepfashion/deepfashion_map.csv")
map_df.set_index('image', inplace=True)

st.title('FashionDiffusion')
st.write('by Soon Yau Cheong')

style_names = ['face', 'hair', 'headwear', 'background', 'top', 'outer', 'bottom', 'shoes', 'accesories']


clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                            std=(0.26862954, 0.26130258, 0.27577711))
clip_transform = T.Compose([
    T.ToTensor(),
    clip_norm])

smpl_image_transform = T.CenterCrop(size=(256, 192))

mask_transform = T.Compose([
    T.Resize(size=[32, 24], interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2. - 1.)
])

image_transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])

lr_transform = T.Compose([
    T.Resize(size=[128, 96], interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Lambda(lambda x: x * 2. - 1.,)])

@st.cache_resource
def upgpt_model(config_file = 'models/upgpt/pt_256/config.yaml',
                ckpt = 'models/upgpt/pt_256/upgpt.pt256.v1.ckpt', 
                device = 'cuda:0'):    
    
    config = OmegaConf.load(config_file)
    model = InferenceModel(config, ckpt, device)

    return model 

model = upgpt_model()
upscale_model = upgpt_model('logs/2023-03-27T09-34-40_x4_upscaling/configs/2023-03-27T09-34-40-project.yaml',
                            'logs/2023-03-27T09-34-40_x4_upscaling/checkpoints/epoch=000002.ckpt', 
                            'cuda:0')

def load_smpl(folder):
    input_mask_type = 'mask'
    pose_path = str(Path(folder)/'pose')
    smpl_image_file = pose_path + '.jpg'
    smpl_file = pose_path + '.p'
    smpl_image = smpl_image_transform(Image.open(smpl_image_file))
    if input_mask_type=='mask':
        mask_file = pose_path + '_mask.png'
        mask_image = Image.open(mask_file)
        person_mask = mask_transform(mask_image)
    elif input_mask_type=='bbox':
        raise "Not supported"
    else:
        person_mask = mask_transform(smpl_image)
       
    with open(smpl_file, 'rb') as f:
        smpl_params = pickle.load(f)
        pred_pose = smpl_params[0]['pred_body_pose']
        pred_betas = smpl_params[0]['pred_betas']
        pred_camera = np.expand_dims(smpl_params[0]['pred_camera'], 0)
        smpl_pose = np.concatenate((pred_pose, pred_betas, pred_camera), axis=1)
        smpl_pose = T.ToTensor()(smpl_pose).view((1,-1))

    return {'smpl':smpl_pose, 
            'smpl_image':smpl_image, 
            'person_mask':person_mask}

def get_styles(input_style_names=style_names):
    style_images = []
    for style_name in input_style_names:
        f_path = local_style_root/f'{style_name}.jpg'

        if f_path.exists():
            style_image = clip_transform((Image.open(f_path)))
        else:
            style_image = clip_norm(torch.zeros(3, 224, 224))
        style_images.append(style_image)
    style_images = torch.stack(style_images)  
    return style_images

left_column, right_column = st.columns([1,3])

# with right_column:
#     f_path = '/home/soon/datasets/deepfashion_multimodal/images/WOMEN-Tees_Tanks-id_00007976-01_4_full.jpg'
#     image = Image.open(f_path)
#     st.image(f_path, width=512)
gen_image = right_column.empty()
show_image_button = right_column.button(label='Show images')
if show_image_button:
    gen_image.image(get_samples(), width=192, caption=[i for i in range(len(get_samples()))])

clear_image_button = right_column.button(label='Clear images')
if clear_image_button:
    clear_image_cache()
    gen_image.empty()

with left_column:
    with st.form(key='input'):    
        st.markdown("##### Text Prompt")
        content_text = st.text_area('Content text', label_visibility='hidden')
        st.markdown("##### Style Text")
        
        style_columns = st.columns(3)

        style_texts = []
        for i, style in enumerate(style_names):
            col = i//3
            with style_columns[col]:
                style_texts.append(st.text_input(style))
        st.markdown("##### Pose")
        pose_ids = [i+1 for i in range(len(pose_images))]
        st.image(pose_images, caption=pose_ids, width=128)
        pose_select = st.radio("Select pose", pose_ids)
        st.markdown("---")
        
        submit_button = st.form_submit_button(label='Generate')
        if submit_button:
            style_features = get_styles()
            batch = {}
            style_texts_dict = dict(zip(style_names, style_texts))
            batch['image'] = image_transform(Image.open(cache_root/'image_256.jpg')) # dummy
            batch['styles'] = model.mix_style(style_features, style_texts_dict)
            batch['txt'] = content_text
            
            batch.update(load_smpl(pose_folders[pose_select - 1]))
            batch = model.create_batch(batch, repeat=1)
            log = model.generate(batch, 200)
            sample = Image.fromarray(np.uint8(log['samples'][0]*255))
            sample.save(local_lowres_root/f'sample_{get_image_number()}.jpg')

            gen_image.image(get_samples(), width=192)


left_2_column, right_2_column = st.columns([1,1])
style_image = right_2_column.empty()

# upscale_button = st.button(label='Upscale')
# if upscale_button:
#     style_features = get_styles(['face'])
#     batch = {}
#     batch['image'] = image_transform(Image.open(cache_root/'image_512.jpg')) # dummy
#     batch['lr'] = lr_transform(Image.open(cache_root/'sample_256.jpg')) # dummy
#     batch['styles'] = model.mix_style(style_features, {})
#     batch['txt'] = ''
#     batch = model.create_batch(batch, repeat=1)
#     log = upscale_model.generate(batch, 200)
#     sample = Image.fromarray(np.uint8(log['samples'][0]*255))
#     gen_image.image(sample, width=384)
#     sample.save(cache_root/'sample_512.jpg')


with left_2_column:
    with st.form("my-form", clear_on_submit=True):
        st.write("Style References")
        style_file = st.file_uploader("Style reference")
        options = None
        if style_file:
            style_local_fname = style_file.name.replace('-','/')
            row = map_df.loc[style_local_fname]
            style_path = row.styles
            bytes_data = style_file.read()
            style_image.image(bytes_data, width=128)
            options = st.multiselect('Select styles', style_names, [])
            
        style_button = st.form_submit_button(label='Get Styles')
        clear_style_button = st.form_submit_button(label='Clear Styles')

        if style_button:
            style_image.empty()
            if options:
                for opt in options:
                    src = styles_root/style_path/f'{opt}.jpg'
                    if src.exists():
                        dst = local_style_root/f'{opt}.jpg'
                        copy(src, dst)
                
            for style in style_names:
                dst = local_style_root/f'{style}.jpg'
                if dst.exists():
                    st.image(Image.open(dst), width=128, caption=style)

        if clear_style_button:
            for style in style_names:
                dst = local_style_root/f'{style}.jpg'
                if dst.exists():
                    os.remove(dst)
