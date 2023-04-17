# UPGPT

### Contribution
Large part of code was adapted from https://github.com/Stability-AI/stablediffusion/. 


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




## BibTeX

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


