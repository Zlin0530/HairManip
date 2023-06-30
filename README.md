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

  - Download StyleGAN model pretrained on FFHQ from [here](https://drive.google.com/file/d/11r6dZpNXrwqWRckIqeXQmQ9NaLDjRfq7/view?usp=sharing), and then place it into the folder `./pretrained_models` .
  - Download Pretrained IR-SE50 model from [here](https://drive.google.com/file/d/1On1Tg0FftyHANycWzIyWwDVpJz1ljuSc/view?usp=sharing), and then place it into the folder `./pretrained_models` .

  

## Testing
- The main testing script is placed in `./mapper/scripts/inference.py`.
- Inference arguments can be found at `./mapper/options/test_options.py`.
- Download our pre-trained HairManip model from [here](https://drive.google.com/file/d/1Wyy46o3yN057rh4BlVDlGziC5jEcg7It/view?usp=sharing), and then place it into the folder `./checkpoints` .
- One test latent code is provided in  `./mapper/text/latents.pt`,  Note that all training and test sets of our model are from CelebA-HQ, if you need to infer other images, please use the [e4e](https://github.com/omertov/encoder4editing) encoder to invert the real images.
### To use text to edit hair color, run the following command

```
python scripts/inference.py --exp_dir=./results --checkpoint_path=../checkpoints/HairManip_model.pt --latents_test_path=./test/latents.pt --editing_type=color --input_type=text --color_description="purple"
```

### To use text to edit hairstyle and reference image to edit hair color, run the following command

```
python scripts/inference.py --exp_dir=./results --checkpoint_path=../checkpoints/HairManip_model.pt --latents_test_path=./text/latents.pt --editing_type=both --input_type=text_image --hairstyle_description="hairstyle_list.txt" --color_ref_img_test_path=text/refimage
```

## Acknowledgements
This code is based on [HairCLIP](https://github.com/wty-ustc/HairCLIP)
