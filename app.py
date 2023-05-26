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

device = 'cuda:0'
styles_root = Path('styles')
cache_root = Path('app_cache')
local_style_root = cache_root/'styles'
local_pose_root = cache_root/'pose'
local_lowres_root = cache_root/'samples_lowres'

os.makedirs(local_style_root, exist_ok=True)
os.makedirs(local_lowres_root, exist_ok=True)
pose_folders = sorted([x[0] for x in os.walk(local_pose_root)][1:])
pose_images = []
for pose_folder in pose_folders:
    pose_images.append(Image.open(glob(os.path.join(pose_folder,'*.jpg'))[0]))
#pose_images = [Image.open(Path(x)/'pose.jpg') for x in pose_folders]

def clear_image_cache():
    for x in glob(str(local_lowres_root/'*')):
        os.remove(x)

def get_image_number():
    image_files = [os.path.split(x)[1] for x in glob(str(local_lowres_root/'*.jpg'))]
    if len(image_files) == 0:
        return 0

    fnames = [f.split('_')[-1].split('.jpg')[0] for f in image_files]
    file_id = max([int(f) for f in fnames if f.isnumeric()])
    #file_id = max([int(f.split('_')[1].split('.jpg')[0]) for f in image_files])
    
    return '{:03d}'.format(file_id + 1)

def get_samples():
    return [Image.open(x) for x in sorted(glob(str(local_lowres_root/'*.jpg')))]

map_df = pd.read_csv("data/deepfashion/deepfashion_map.csv")
map_df.set_index('image', inplace=True)

st.title('UPGPT - mix-and-match text and visual prompts for human image generation')
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
    T.Pad((4,0),padding_mode='edge'),
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

model = upgpt_model(device=device)

disable_upscale = True
upscale_ckpt = "models/upgpt/upscale/upgpt.upscale.v1.ckpt"
if os.path.exists(upscale_ckpt):
    upscale_model = upgpt_model('models/upgpt/upscale/config.yaml',
                                upscale_ckpt, 
                                device)
    disable_upscale = False
def load_smpl(folder):
    smpl_file = glob(str(Path(folder)/'*.p'))[0]
    smpl_image_file = glob(str(Path(folder)/'*.jpg'))[0]
    input_mask_type = 'mask'
    #pose_path = str(Path(folder)/'pose')
    #smpl_image_file = pose_path + '.jpg'
    #smpl_file = pose_path + '.p'
    smpl_image = smpl_image_transform(Image.open(smpl_image_file))
    if input_mask_type=='mask':
        mask_file = glob(str(Path(folder)/'*.png'))[0]
        #mask_file = pose_path + '_mask.png'
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


left_column, mid_column, right_column, sr_column = st.columns([1,1,1,1])
right_column.markdown("##### Generated Images")
gen_image = right_column.empty()
low_res_images = get_samples()
image_ids = [i+1 for i in range(len(low_res_images))]     


def display_samples():
    global image_ids
    low_res_images = get_samples()
    image_ids = [i+1 for i in range(len(low_res_images))]
    gen_image.image(low_res_images, width=192, caption=image_ids)

display_samples()



with left_column:
    with st.form(key='input'):    
        st.markdown("##### Text Prompt")
        default_text = "a woman is wearing a sleeveless tank and a short skirt."
        content_text = st.text_area('Content text', label_visibility='hidden', value=default_text)
        st.markdown("##### Style Text")
        
        style_columns = st.columns(3)

        style_texts = []
        for i, style in enumerate(style_names):
            col = i//3
            with style_columns[col]:
                style_texts.append(st.text_input(style))
        st.markdown("##### Pose")
        pose_ids = [i+1 for i in range(len(pose_images))]
        st.image(pose_images, caption=pose_ids, width=96)
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
            display_samples()
            # gen_image.image(get_samples(), width=192)
            # low_res_images = get_samples()
            # image_ids = [i+1 for i in range(len(low_res_images))]

with mid_column:
    #left_2_column, right_2_column = st.columns([1,1])
    #style_image = right_2_column.empty()
    #with left_2_column:
    with st.form("my-form", clear_on_submit=False):
        st.markdown("##### Image Styles")
        style_file = st.file_uploader("Style reference")
        style_image = st.empty()
        options = None
        if style_file is not None:
            style_local_fname = style_file.name.replace('-','/')
            row = map_df.loc[style_local_fname]
            style_path = row.styles
            bytes_data = style_file.read()
            style_image.image(bytes_data, width=128)
            options = st.multiselect('Select styles', style_names, [])
            style_file = None
        style_button = st.form_submit_button(label='Show/Get Styles')
        #clear_style_button = st.form_submit_button(label='Clear Styles')

        if style_button:
            #style_image.empty()
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

        styles_to_delete = []   
        for style in style_names:
            dst = local_style_root/f'{style}.jpg'
            if dst.exists():
                styles_to_delete.append(style)                
                #os.remove(dst)

        del_options = st.multiselect('Select styles to delete', styles_to_delete, styles_to_delete)
        del_style_button = st.form_submit_button(label='Remove Styles')
        if del_style_button:
            for style in del_options:
                dst = local_style_root/f'{style}.jpg'
                os.remove(dst)

                

with right_column:
    
    show_image_button = right_column.button(label='Show images')

    if show_image_button:
        display_samples()

    image_files = sorted(glob(str(local_lowres_root/'*.jpg')))
    delete_ids = [i+1 for i in range(len(image_files))]
    del_options = st.multiselect('Select images to delete', delete_ids, [])
    clear_image_button = st.button(label='Delete images')
    if clear_image_button:
        for del_option in del_options:
            os.remove(image_files[del_option-1])
        #clear_image_cache()
        #gen_image.empty()
    display_samples()

with sr_column:    
    st.markdown('#####  Upscale')
    upscale_select = st.selectbox('Upscale', image_ids, label_visibility='hidden')
    clear_image_button = st.button(label='Upscale', disabled=disable_upscale)
    if clear_image_button:
        low_res_images = get_samples()
        style_features = get_styles(style_names)
        batch = {}
        batch['image'] = image_transform(Image.open(cache_root/'image_512.jpg')) # dummy
        lr_image = low_res_images[upscale_select - 1]
        batch['lr'] = lr_transform(lr_image)
        batch['styles'] = model.mix_style(style_features, {})
        batch['txt'] = content_text
        batch = model.create_batch(batch, repeat=1)
        log = upscale_model.generate(batch, use_ema=False)
        sample = Image.fromarray(np.uint8(log['samples'][0]*255))
        st.image(sample, width=384)
        fname = cache_root/'sample_512.png'
        sample.save(fname)
        with open(str(fname), "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="download.png",
                    mime="image/png"
                )