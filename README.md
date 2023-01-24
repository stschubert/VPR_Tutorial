# VPR_Tutorial
This repository provides the example code from our ongoing work about a tutorial for visual place recognition (VPR) and is work in progress.
The code performs VPR on the GardensPoint day_right--night_right dataset. Output is a plotted pr-curve and the AUC performance, as shown below.

## How to run the code
```
python3 demo.py
```
The GardensPoints Walking dataset will be downloaded automatically.

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
