# swarmingAnalysis

## introduction

This is a repo aiming to analysis bacteria swarming, especially the characteristics of the bacteria swarm. The overall pipeline of the experiment is:

1. Do the experiment, and take the photos as datasets
2. Pre-process the datasets
3. Analyze the datasets:
   1. Segment all of the parts of the image
   2. Select the parts of agar plate and the swarm
   3. Calculate the ratios of the swarm and the agar plate
   4. Analyze the shape of the swarm
4. Use it to generate the formula!

## environment setup

### hardware environment

NVIDIA GPU with >= 8GB memory for images below 10MB.

with >=16GB memory for images taken by industrial camera.

### python environment

- python >= 3.8
- [pytorch](https://pytorch.org/) >= 1.12
- CUDA and CuDNN installed
- opencv-python installed
- segment-anything installed (via pip)

### semantic segmentation model

In this experiment, [segment-anything](https://github.com/facebookresearch/segment-anything) is used for the model. You can download the model weight file `vit_h` in the github repo, and try the demo of the model at its [official website](https://segment-anything.com/demo).

## overall structures

- `1.sh`: the script for running the `amg.py`
- `amg.py`: sample code for running the model for segment anything in the image
  - `input`: the folder for the input images of `amg.py`
  - `output`: the folder for the output images of `amg.py`
- `automatic_mask_generator_example.ipynb`: the jupyter notebook for running the model for generating the mask using several points and rectangles input.
- `predictor_example.ipynb`: the jupyter notebook for running the model for segment anything in the image.
- `swarming-pipeline.ipynb`: the jupyter notebook for my previous try of the model(failed)
- `graphical.py`: opensource python script for automatic mask generator using a rectangle
- `hough.py`: the python script for generating the mask for agar plate using hough transformation
- `pngAnalysis.py`: the python script used for analyzing the shape and numbers of pixels of the mask

## TODO list

- [ ] automatically classify the agar plate and the swarm in these images
- [ ] add more accuracy while classifying the swarm of vertical images
- [ ] Calculate the IoU of the model between that and our own mask labelled using `labelme` previously
