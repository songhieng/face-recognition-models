# Face Recognition Models

This repository contains implementations of multiple face recognition models: ElasticFace, AdaFace, FaceNet512, and Buffalo_L.

## Structure

```
.
├── models/                     # Main directory
│   ├── __init__.py             # Package initialization
│   ├── elasticface/            # ElasticFace model
│   │   ├── __init__.py         # Model initialization
│   │   ├── backbone.py         # iResNet50 backbone implementation
│   │   ├── model.py            # ElasticFace model implementation
│   │   └── benchmark.py        # Benchmark utilities for CFP dataset
│   ├── adaface/                # AdaFace model
│   │   ├── __init__.py         # Model initialization
│   │   ├── backbone.py         # iResNet101 backbone implementation
│   │   ├── model.py            # AdaFace model implementation
│   │   └── benchmark.py        # Benchmark utilities for CFP dataset
│   ├── facenet512/             # FaceNet512 model
│   │   ├── __init__.py         # Model initialization
│   │   └── model.py            # FaceNet512 model implementation
│   └── buffalo_l/              # Buffalo_L model
│       ├── __init__.py         # Model initialization
│       └── model.py            # Buffalo_L model implementation
├── pretrain-model/             # Directory for pretrained model weights
│   ├── 295672backbone.pth      # Pretrained ElasticFace backbone weights (needs to be downloaded)
│   ├── adaface_ir101_webface12m.ckpt  # Pretrained AdaFace model weights (needs to be downloaded)
│   ├── buffalo_l/              # Buffalo_L model files (downloaded automatically)
│   └── models/                 # Additional model files (downloaded automatically)
├── test/                       # Directory for test images
│   ├── out.jpg                 # Test image 1
│   └── out1.png                # Test image 2
├── benchmark_results/          # Directory for benchmark results
├── backup_scripts/             # Directory for backup of unused scripts
├── test_models.py              # Unified script to test all face recognition models
├── benchmark_all_models.py     # Unified benchmarking script for all models
├── benchmark_cfp.py            # Original CFP dataset benchmark script
├── visualize_benchmark_results.py # Script for visualizing benchmark results
├── run_benchmarks.py           # Main script to run all benchmarks
├── compare_all_models.py       # Script to compare all models side-by-side
├── create_demo_model.py        # Script to create demo models
├── cleanup.py                  # Script to remove redundant files
├── SUMMARY.md                  # Summary of model comparison results
├── BENCHMARK_README.md         # Documentation for benchmarking system
├── pretrain-model-readme.md    # Information about obtaining model files
└── requirements.txt            # Dependencies
```

## Requirements

```
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.19.2
Pillow>=8.0.0
scikit-learn>=0.24.0
pandas>=1.2.0
tqdm>=4.54.0
matplotlib>=3.3.0
deepface>=0.0.75
insightface>=0.6.0
opencv-python>=4.5.0
tensorflow>=2.4.0
```

## Pretrained Model Files

Due to their large size, pretrained model files are not included in this repository. Please see `pretrain-model-readme.md` for instructions on how to download and set up the model files.

## Usage

### Testing Face Recognition Models

To test all models at once:

```bash
python test_models.py --model all --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

To test a specific model:

```bash
python test_models.py --model elasticface --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

Available model options: `elasticface`, `adaface`, `facenet512`, `buffalo_l`, or `all`

### Benchmarking on the CFP Dataset

For comprehensive benchmarking of all models:

```bash
python run_benchmarks.py --data_dir path/to/cfp-dataset/Data/Images
```

This will:
1. Run benchmarks on all models
2. Generate metrics and CSV results
3. Create visualizations
4. Generate an HTML report

For more detailed information on benchmarking, see `BENCHMARK_README.md`.

### Comparing All Models Side-by-Side

```bash
python compare_all_models.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

This will:
1. Run all models on the same image pair
2. Compare their similarity scores, match results, and processing times
3. Create a comprehensive visualization saved to `results/all_models_comparison.png`

### Using Demo Models

For demonstration purposes when pretrained models are unavailable, you can use the `--use_demo` flag:

```bash
python test_models.py --model elasticface --image1 test/out.jpg --image2 test/out1.png --use_demo
python run_benchmarks.py --data_dir path/to/cfp-dataset/Data/Images --use_demo
```

To create the demo models manually:
```bash
python create_demo_model.py
```

## Notes

- The ElasticFace model performs better with a threshold around 0.3
- The AdaFace model performs better with a threshold around 0.4
- The FaceNet512 model performs better with a threshold around 0.4
- The Buffalo_L model performs better with a threshold around 0.5
- For best results, use aligned face images
- All models are robust to pose variations, making them suitable for comparing frontal and profile face images
- AdaFace uses a more advanced IR-101 backbone and generally achieves better performance than ElasticFace
- The AdaFace implementation includes all the necessary code, but the actual pretrained model weights need to be obtained separately. For accurate results, please download the proper `adaface_ir101_webface12m.ckpt` file and place it in the `pretrain-model` directory
- The FaceNet512 and Buffalo_L models use libraries (DeepFace and InsightFace) that download model weights automatically

## Model Architecture

### ElasticFace
- Backbone: ResNet-50
- Features: 512-dimensional embedding
- Loss function: CosFace or SphereFace

### AdaFace
- Backbone: IR-101 (Improved Residual Network with 101 layers)
- Features: 512-dimensional embedding
- Adaptive margin mechanism that adjusts based on feature quality

### FaceNet512
- Implementation from DeepFace library
- Features: 512-dimensional embedding
- Google FaceNet architecture with expanded embedding size

### Buffalo_L
- Implementation from InsightFace library
- Features: 512-dimensional embedding
- Large model variant of InsightFace with improved accuracy

## See Also

Check out `SUMMARY.md` for a detailed comparison of all models, including performance metrics and implementation details. 