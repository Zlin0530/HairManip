# HairManip：High Quality Hair Manipulation via Hair Element Disentangling
![图1](https://github.com/Zlin0530/HairManip/blob/main/images/fig1.jpg)

## Overview
> This repository hosts the official PyTorch implementation of the paper:  
>  "**HairManip：High Quality Hair Manipulation via Hair Element Disentangling**".  

## News
**`2023.06.24`**: We will release our testing code and pretrained model.   
  To be continued...

## Dependences
### Prerequisites
```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install tensorflow-io
```

- Download pre-trained models :

  - Download our pre-trained HairManip model from [here](https://drive.google.com/file/d/1Wyy46o3yN057rh4BlVDlGziC5jEcg7It/view?usp=sharing), and then place it into the folder `./checkpoints` .
  - Download the CelebA-HQ test set latent codes from [here]().

  

## Testing



## Acknowledgements
This code is based on [HairCLIP](https://github.com/wty-ustc/HairCLIP)
