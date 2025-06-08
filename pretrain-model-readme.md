# Pretrained Model Files

This directory should contain the following model files which are not included in the repository due to their large size:

## Model Files
- `adaface_ir101_webface12m.ckpt` (1.4GB) - AdaFace model weights
- `295672backbone.pth` (249MB) - ElasticFace backbone weights
- `buffalo_l/` - Buffalo_L model files
- `models/` - Additional model files

## How to Obtain Model Files

### AdaFace
Download the AdaFace model weights from the official repository:
- https://github.com/mk-minchul/AdaFace

### ElasticFace
Download the ElasticFace model weights from the official repository:
- https://github.com/fdbtrs/ElasticFace

### Buffalo_L and FaceNet512
These models are automatically downloaded when you run the scripts that use them, as they use the InsightFace and DeepFace libraries respectively.

## Directory Structure
After downloading, your directory structure should look like:
```
pretrain-model/
├── adaface_ir101_webface12m.ckpt
├── 295672backbone.pth
├── buffalo_l/
│   └── (model files)
└── models/
    └── (model files)
``` 