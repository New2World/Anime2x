# Anime2x

This is a deep neural network model for scaling anime illustrations by **2x**

## Model

The framework of this model is based on [**F**ast **S**uper **R**esolution **CNN**](https://arxiv.org/abs/1608.00367). However, different from origin FSRCNN, the first part is replaced by multiple convolutional layers and the kernel size is reduced from 5 to 3.  
The scale factor is fixed to 2 in this model, as it's mentioned in the paper that if we need to change the factor, the only thing we need to do is to change the stride in deconvolutional layer and fine-tune the last part.

## Train

The training images are collected by myself from [pixiv.net](https://www.pixiv.net/), and over 30 authors' works (5,000 illustrations) are included.  
Before fed into this model, a 224x224 image patch will be randomly cropped from each origin data. All data will be flipped horizontally and vertically randomly to do data augmentation.

## Run

To run this model:

```bash
$ python3 solver.py -i <input image> -o <output path/image>
```
