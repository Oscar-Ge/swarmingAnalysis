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

## usage

The analysis workflow consists of three main steps:

### 1. Run SAM segmentation (1.sh)

First, configure the input directory in `1.sh` to point to your image folder, then run:

```bash
bash 1.sh
```

This script will:
- Run the Segment Anything Model (SAM) on all images in the specified directory
- Generate segmentation masks for each image
- Package the output into a zip file named after the input folder
- Clean up the output directory

### 2. Analyze segmentation results (analysis.py)

After segmentation, run the analysis script:

```bash
python analysis.py
```

This script will:
- Display all segmentation masks for each experiment
- Prompt you to manually select which mask corresponds to the agar plate and which to the bacterial colony
- Calculate metrics including:
  - Area of plates and colonies
  - Ellipticity (major axis / minor axis) of colonies
  - Area ratio of colony to plate
- Save results to two CSV files:
  - `plates_data.csv`: agar plate information
  - `colonies_data.csv`: colony information with calculated metrics

### 3. Visualize results (visualize_sam_results.py)

Finally, generate visualizations of the segmentation results:

```bash
# Interactive mode - select and visualize a single high-quality sample
python visualize_sam_results.py

# Batch mode - generate visualizations for top N samples
python visualize_sam_results.py batch 5

# Batch mode with direction filter
python visualize_sam_results.py batch 5 up
```

This script will:
- Read colony data from the CSV file
- Create three-panel visualizations showing:
  - Original image
  - Segmentation overlay with bounding box
  - Segmentation mask
- Display metadata including date, direction, area, ellipticity, and IOU score
- Save visualization images to `visualization_output/` directory

## overall structures

- `1.sh`: bash script for running SAM segmentation via `amg.py`
- `amg.py`: sample code for running the Segment Anything Model
  - `input`: the folder for the input images of `amg.py`
  - `output`: the folder for the output images of `amg.py`
- `analysis.py`: interactive script for analyzing SAM results and extracting plate/colony data
- `visualize_sam_results.py`: script for generating visualizations of segmentation results
- `automatic_mask_generator_example.ipynb`: the jupyter notebook for running the model for generating the mask using several points and rectangles input
- `predictor_example.ipynb`: the jupyter notebook for running the model for segment anything in the image
- `swarming-pipeline.ipynb`: the jupyter notebook for previous experiments with the model
- `graphical.py`: opensource python script for automatic mask generator using a rectangle
- `hough.py`: the python script for generating the mask for agar plate using hough transformation
- `pngAnalysis.py`: the python script used for analyzing the shape and numbers of pixels of the mask

## TODO list

- [ ] automatically classify the agar plate and the swarm in these images
- [ ] add more accuracy while classifying the swarm of vertical images
- [ ] Calculate the IoU of the model between that and our own mask labelled using `labelme` previously
