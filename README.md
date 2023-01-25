# Visual Place Recognition: A Tutorial
Work in progress: This repository provides the example code from our ongoing work on a tutorial for visual place recognition (VPR).
The code performs VPR on the GardensPoint day_right--night_right dataset. Output is a plotted pr-curve and the AUC performance, as shown below.


## How to run the code
```
python3 demo.py
```
The GardensPoints Walking dataset will be downloaded automatically.


### Expected output
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


## List of existing open-source implementations for VPR
[//]: # (use <td colspan=3> or rowspan to combine cells)
<table>
    <thead>
        <tr>
            <th>Method</th>
            <th>Paper</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://www.tu-chemnitz.de/etit/proaut/en/research/prstructure.html">ICM</a></td>
            <td><a href="http://doi.org/10.15607/RSS.2021.XVII.091">paper</a></td>
            <td>Graph optimization of the similarity matrix S</td>
        </tr>
        <tr>
            <td><a href="https://www.di.ens.fr/willow/research/netvlad/">NetVLAD</a></td>
            <td><a href="https://doi.org/10.1109/CVPR.2016.572">paper</a></td>
            <td>Holistic descriptor</td>
        </tr>
        <tr>
            <td><a href="url">method</a></td>
            <td><a href="url">paper</a></td>
            <td></td>
        </tr>
    </tbody>
</table>


