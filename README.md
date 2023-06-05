# Breaking Change (1st Jun 2023)
I have updated the code for pose interpolation. However, you will need to download the new model file interp_256.zip (previously pt_256.zip). The app now also come with pre-loaded style images and generated examples.

# UPGPT
This is the official Github repo for the paper "UPGPT: Universal Diffusion Model for Person Image Generation, Editing and Pose Transfer"
https://arxiv.org/abs/2304.08870
![banner](https://user-images.githubusercontent.com/19167278/234025496-242e3df0-5f5c-49bc-ba08-9aeaa5907172.png)

The code was adapted from https://github.com/Stability-AI/stablediffusion/.

![App](./assets/app.png)

## Video Demo (HD) 

[![Video Demo (HD)](assets/video.jpg)](https://youtu.be/2E8MSRlcN54)


BibTeX:
```
@misc{upgpt,
      title={UPGPT: Universal Diffusion Model for Person Image Generation,
Editing and Pose Transfer}, 
      author={Soon Yau Cheong and Armin Mustafa and Andew Gilbert},
      year={2023},
      journal={arXiv:2304.08870},
      primaryClass={cs.CV}
}
```

## To-Do
- [x] release model weights
- [x] release inference app
- [x] release interpolation model
- [x] release SMPL data and training script

## Paper's Result
The ground truth and generated images used in the paper can be downaloded from
[the repo release.](https://github.com/soon-yau/upgpt/releases/tag/v1.0.0)

## Requirements
A suitable [conda](https://conda.io/) environment named `upgpt` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate upgpt
```
## Files
Model checkpoints and dataset can be downloaded from [HuggingFace](https://huggingface.co/soonyau/upgpt/tree/main). 

## App
This demonstration uses pre-segmented style images from DeepFashion Multimodal dataset and does not support arbitrary images that you upload. We provide a few samples that you can play with in the app.

- Download models interp_256.zip and upscale.zip(optional) and unzip into ./models/upgpt
- Start the app by typing in terminal `streamlit run app.py`
- Click "Image Styles->Browse files" to select images from ./fashion. Then "select styles" and click "Show/Get Styles" to extract style images. The model is trained for pose transfer, hence a face style image is advised to produce good result.
- Entering "style text" will override corresponding style images, therefore remove style text if you want to use style image.
 
### Additional data
1. Download and unzip deepfashion_inshop.zip into datasets/deepfashion_inshop.
2. You can try more style images from the DeepFashion Multimodal dataset by downloading and unzip images.zip from [DeepFashion Multimodal dataset](https://github.com/yumingj/DeepFashion-MultiModal). Use this inplace of ./fashion to select fashion images from. Also, use 'rm -r app_cache/styles && ln -s deepfashion_inshop/styles app_cache/styles' to link to the full dataset style images. 
 
## Training
Followed [1] to download data; and run train.sh or 
```python main.py -t --base configs/deepfashion/bbox.yaml --gpus 0, --scale_lr False --num_nodes 1```
Checkpoints and generated images will be saved in ./logs.



