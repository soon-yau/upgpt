# UPGPT
This is the official Github repo for the paper "UPGPT: Universal Diffusion Model for Person Image Generation, Editing and Pose Transfer"
https://arxiv.org/abs/2304.08870

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

Demo Video:

https://user-images.githubusercontent.com/19167278/233998033-7bfbeec5-e144-4928-b2ed-82f8b52c463c.mp4


## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2
pip install -e .
```







