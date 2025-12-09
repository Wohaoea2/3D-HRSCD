# 3D-HRSCD

Official Pytorch Code base for "3D-HRSCD: Exploiting the Potential of Multi-scale Features by 3D Convolution"

[Project](https://github.com/Wohaoea2/3D-HRSCD)

## Introduction

Our study aims to develop an effective multi-task network for SCD by modeling temporal dependency. We propose 3D-HRSCD, a novel architecture that utilizes 3D convolution to model temporal dependency across HRNet’s multi-resolution features. The core of this architecture is 3D Convolution Fusion Oriented to Multiscale Features (3DFOM) module, which makes adequate interaction in channel, spatial and temporal dimensions across multiscale features. To support more efficient temporal dependency modeling in 3DFOM, Cosine Similarity-based Temporal Multi-Scales Attention (CTMA) module serves as a preprocessing stage by enhancing features in change regions. Additionally, Comprehensive Semantic Consistency (CSC) loss function is introduced to further suppress pseudo-changes and reduce semantic recognition errors. 

<!-- <p align="center">
  <img src="imgs/flowchart.png" width="800"/>
</p> -->

## Using the code:

The code is stable while using Python 3.8, torch 2.2.2, CUDA 12.1

- Clone this repository:
```bash
git clone https://github.com/Wohaoea2/3D-HRSCD
```

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── im1
    |   ├── 00003.png
    │   ├── 00013.png
    │   ├── 00015.png
    │   ├── ...
    |
    └── im2
    |   ├── 00003.png
    │   ├── 00013.png
    │   ├── 00015.png
    |   ├── ...
    └── label1
    |   ├── 00003.png
    │   ├── 00013.png
    │   ├── 00015.png
    |   ├── ...
    └── label2
    |   ├── 00003.png
    │   ├── 00013.png
    │   ├── 00015.png
    └── ├── ...
```

For validation and testing datasets, the same structure as the above.

## Training and testing

1. Train the model.
```
run train.SCD.py
```
2. Predict the SCD results.
```
run pred_SCD.py
```
### Semantic change detection datasets: 

SECOND dataset: https://drive.google.com/file/d/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc/view

### Citation:
If you find this work useful or interesting, please consider citing the following reference.
```
[1] Y. Song, S. Fang, Z. Li, S. Wang and E. Zhao, "3D-HRSCD: Exploiting the Potential of Multi-scale Features by 3D Convolution," in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2025.3591276. 

