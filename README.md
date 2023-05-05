# UPGPT
This is the official Github repo for the paper "UPGPT: Universal Diffusion Model for Person Image Generation, Editing and Pose Transfer"
https://arxiv.org/abs/2304.08870
![banner](https://user-images.githubusercontent.com/19167278/234025496-242e3df0-5f5c-49bc-ba08-9aeaa5907172.png)

## Video Demo (HD) 

[![Video Demo (HD)](assets/video.jpg)](https://youtu.be/2E8MSRlcN54)


BibTeX:
```
@misc{upgpt,
      title={UPGPT: Universal Diffusion Model for Person Image Generation,
Editing and Pose Transfer}, 
      author={Soon Yau Cheong and Armin Mustafa and Andew Gilbert},
      year={2023},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



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

## App
As the SMPL files are not optimised and their size are too big to be hosted online, so we will not be providing them for training nor inference. However, we provide a few samples that you can play with in the app.

Download files from [Google Drive](https://drive.google.com/drive/folders/1ifKoQEOir9NXmZGrPSIYpFT5L4pSHTBh?usp=share_link):
- Unzip style.zip 
- Unzip models pt_256.zip and upscale.zip into ./models/upgpt
- Start the app by typing in terminal `streamlit run app.py`


<!---  https://user-images.githubusercontent.com/19167278/233998033-7bfbeec5-e144-4928-b2ed-82f8b52c463c.mp4 --->


