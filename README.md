# VPR_Tutorial
Code for the VPR tutorial that performs VPR on the GardensPoint day_right--night_right dataset. Output is a plotted pr-curve and the AUC performance.

## How to run the code
```
python3 demo.py
```
The GardensPoints Walking dataset will be downloaded automatically.

## Requirements
- matplotlib==3.1.2
- numpy==1.17.4
- Pillow==9.1.0
- scipy==1.3.3
- tensorflow==2.8.0
- tensorflow_hub==0.12.0

## Currently implemented:
- download and load images
- compute DELF descriptors
- run HDC for feature aggregation into holistic descriptors
- perform full descriptor comparison to obtain similarity matrix S
- evaluation (precision-recall curve and AUC)

## List of existing open-source implementations
