# SQCAM: Distribution-Aware Channel Recalibration for Dermoscopic Lesion Segmentation

Official implementation of **SQCAM (Soft Quantile Channel Attention Module)** for dermoscopic skin lesion segmentation.

This repository provides the **source code**, **training pipeline**, and **pretrained model weights** of the paper:

> **SQCAM: SQCAM: Soft Quantile Channel Attention Module for Skin Lesion Segmentation**  
> *(paper submitted)*

---

The method is evaluated on multiple architectures :
- U-Net
- U-Net++
- DeepLabv3+
- ResNet18-FPN
- SegFormer-B0

and Datasets:
- ISIC 2018
- PH2

Datasets (ISIC 2018, PH2) are publicly available from their official sources
and are not redistributed in this repository.

---

## ✨ Features

- ✅ Plug-and-play attention module
- ✅ Cross-architecture compatibility
- ✅ Cross-dataset evaluation
- ✅ Reproducible training pipeline
- ✅ Pretrained model weights provided

---

## 📁 Repository Structure
> *(For clarity the repository is split to two parts)*
         
SQCAM/
│
├── Attention mechanisms comparaisons/
│   ├── dataclass.py
│   ├── Test.py
│   ├── Train.py
│   └── Models/
│       ├── UNet.py
│       ├── UNet_CBAM.py
│       ├── UNet_ECA.py
│       ├── UNet_SE.py
│       ├── UNet_SQCAM.py
│       └── Pretrained - weights/
│           ├── UNet/
│           │   ├── metrics.csv
│           │   ├── UNet.png
│           │   └── unet_model.pth
│           ├── UNet_CBAM/
│           │   ├── metrics.csv
│           │   ├── UNet_CBAM.png
│           │   └── unet_model.pth
│           ├── UNet_ECA/
│           │   ├── metrics.csv
│           │   ├── UNet_ECA.png
│           │   └── unet_model.pth
│           ├── UNet_SE/
│           │   ├── metrics.csv
│           │   ├── UNet_SE.png
│           │   └── unet_model.pth
│           └── UNet_SQCAM/
│               ├── metrics.csv
│               ├── UNet_SQCAM.png
│               └── unet_model.pth
│
├── Cross-Architecture generalization/
│   ├── dataclass.py
│   ├── test.py
│   ├── train.py
│   ├── train (UNetpp).py
│   └── Models/
│       ├── deeplabv3plus.py
│       ├── Resnet18_FPN.py
│       ├── SegFormer_B0.py
│       └── unetPP.py
│       └── Pretrained - weights/
│           ├── Deeplabv3plus/
│           │   ├── link to model weights (.pth file).txt
│           │   └── metrics.csv
│           ├── Deeplabv3plus SQCAM/
│           │   ├── link to model weights (.pth file).txt
│           │   └── metrics.csv
│           ├── ResNet18_FPN/
│           │   ├── link to model weights (.pth file).txt
│           │   └── metrics.csv
│           ├── ResNet18_FPN SQCAM/
│           │   ├── link to model weights (.pth file).txt
│           │   └── metrics.csv
│           ├── SegFormerB0/
│           │   ├── metrics.csv
│           │   └── model.pth
│           ├── SegFormerB0 SQCAM/
│           │   ├── metrics.csv
│           │   └── model.pth
│           ├── UNetpp/
│           │   ├── metrics.csv
│           │   └── model.pth
│           └── UNetpp SQCAM/
│               ├── metrics.csv
│               └── model.pth
│
└── README.md
 
