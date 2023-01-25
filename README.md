# VPR_Tutorial
Work in progress: This repository provides the example code from our ongoing work on a tutorial for visual place recognition (VPR).
The code performs VPR on the GardensPoint day_right--night_right dataset. Output is a plotted pr-curve and the AUC performance, as shown below.

## How to run the code
```
python3 demo.py
```
The GardensPoints Walking dataset will be downloaded automatically.

### Expected Output
You should get an output similar to this:
```
python3 demo.py
===== Load dataset
===== Load dataset GardensPoint day_right--night_right
===== Compute local DELF descriptors
===== Compute holistic HDC-DELF descriptors
===== Compute cosine similarities S
===== Match images
===== Evaluation

===== AUC (area under curve): 0.7419284432277942 
```

| Precision-recall curve | Matchings M | Examples for a true positive and a false positive |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="output_images/pr_curve.jpg" alt="precision-recall curve P=f(R)" height="200" width="auto">  |  <img src="output_images/matchings.jpg" alt="output_images/matchings.jpg" height="200" width="auto"> | <img src="output_images/examples_tp_fp.jpg" alt="Examples for true positive (TP) and false positive (FP)" height="200" width="auto">| 

## Requirements
The code was tested with the following library versions:
```
pipreqs VPR_Tutorial/ --print
```
- matplotlib==3.1.2
- numpy==1.17.4
- Pillow==9.4.0
- scikit_image==0.19.3
- scipy==1.3.3
- skimage==0.0
- tensorflow==2.10.0
- tensorflow_hub==0.12.0


## Currently implemented
- download and load images
- compute DELF descriptors
- run HDC for feature aggregation into holistic descriptors
- perform full descriptor comparison to obtain similarity matrix S
- match images using the best match per query or an automatic thresholding
- evaluation (precision-recall curve and AUC)

## List of existing open-source implementations
- ...
