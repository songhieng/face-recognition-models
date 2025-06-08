# Face Recognition Model Benchmarking

This benchmarking system evaluates the performance of multiple face recognition models on the CFP (Celebrities in Frontal-Profile) dataset.

## Models Evaluated

1. **ElasticFace** - Custom implementation with ResNet-50 backbone
2. **AdaFace** - Advanced model with IR-101 backbone and adaptive margin
3. **FaceNet512** - DeepFace library's FaceNet implementation
4. **Buffalo_L** - InsightFace library's Buffalo_L model

## Requirements

See `requirements.txt` for all dependencies. The benchmark scripts leverage GPU acceleration via CUDA when available.

## Dataset Structure

The benchmark expects the CFP dataset in the following structure:

```
cfp-dataset/
└── Data/
    └── Images/
        ├── 001/
        │   ├── frontal/
        │   │   ├── 01.jpg
        │   │   └── ...
        │   └── profile/
        │       ├── 01.jpg
        │       └── ...
        ├── 002/
        │   ├── frontal/
        │   └── profile/
        └── ...
```

Each person's directory contains two subdirectories: `frontal` and `profile`.

## Running Benchmarks

### Quick Start

To run the complete benchmark on all models:

```bash
python run_benchmarks.py --data_dir path/to/cfp-dataset/Data/Images
```

This will:
1. Run the benchmarks on all models
2. Generate visualizations
3. Create an HTML report with the results

### Options

```bash
python run_benchmarks.py --data_dir path/to/cfp-dataset/Data/Images --models elasticface,adaface --output_dir custom_results --use_demo
```

Parameters:
- `--data_dir`: Path to the CFP dataset images directory
- `--output_dir`: Directory to save results (default: benchmark_results)
- `--models`: Comma-separated list of models to test (elasticface,adaface,facenet512,buffalo_l) or "all"
- `--use_demo`: Use demo models when the original pretrained models are unavailable

## Benchmark Metrics

The benchmark calculates and reports the following metrics:

1. **Verification Accuracy**:
   - Frontal-Frontal (FF): Accuracy between frontal face images
   - Profile-Profile (PP): Accuracy between profile face images
   - Frontal-Profile (FP): Accuracy between frontal and profile images

2. **ROC Metrics**:
   - ROC Curve and AUC (Area Under Curve)
   - EER (Equal Error Rate)
   - Optimal threshold that minimizes FAR and FRR

3. **Processing Time**: Time taken for face recognition processing

4. **Similarity Distributions**: Statistical analysis of similarity scores

## Individual Scripts

### 1. benchmark_all_models.py

Runs the benchmark on selected models and generates CSV result files.

```bash
python benchmark_all_models.py --data_dir path/to/dataset --models all
```

### 2. visualize_benchmark_results.py

Creates visualizations from benchmark results:

```bash
python visualize_benchmark_results.py --benchmark_dir benchmark_results
```

### 3. run_benchmarks.py

The main script that orchestrates the entire benchmark process.

## Output

The benchmark process generates:

1. **CSV Files**:
   - `model_comparison_summary.csv`: Summary of all model results
   - `{model}_person_results.csv`: Per-person results for each model
   - `{model}_verification_metrics.csv`: Verification metrics at different thresholds

2. **Visualizations**:
   - ROC curves
   - Accuracy comparisons
   - Similarity distributions
   - Performance radar chart
   - Processing time comparison

3. **HTML Report**: `benchmark_report.html` containing all results and visualizations

## Adding New Models

To add a new model to the benchmark:

1. Implement the model in the `models/` directory
2. Add model loading and embedding extraction functions
3. Update `benchmark_all_models.py` to include the new model

## Tips for Best Results

- Use a GPU for faster processing
- Ensure face images are properly aligned
- For large datasets, you can sample a subset of images for quicker testing
- Use the `--use_demo` flag when pretrained model weights are unavailable

## Windows Compatibility

The benchmark scripts are compatible with Windows systems. Here are some tips for Windows users:

1. If you encounter Unicode encoding errors, the updated scripts should handle them correctly
2. For better performance on Windows, consider using Anaconda or Miniconda to manage your Python environment
3. Make sure to install all dependencies with the correct versions for your system

## Troubleshooting

### Missing Dependencies

If you encounter errors about missing dependencies, run:

```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues

If you're having issues with CUDA:

1. Verify you have a compatible NVIDIA GPU
2. Ensure you have the correct CUDA toolkit installed for your PyTorch version
3. You can still run the benchmarks without CUDA by using CPU:

```bash
python run_benchmarks.py --data_dir path/to/dataset --use_demo
```

### Buffalo_L Model Path

On Windows systems, the Buffalo_L model path should be specified with backslashes:

```
pretrain-model\buffalo_l
```

The scripts have been updated to handle path separators correctly on all platforms, but if you encounter issues with the Buffalo_L model not being found, make sure the path is correct for your platform.

### Memory Issues

If you encounter memory errors:

1. Test with a smaller subset of the dataset
2. Run benchmarks for one model at a time:

```bash
python run_benchmarks.py --data_dir path/to/dataset --models elasticface
```

### Visualization Errors

If visualization generation fails:

```bash
python run_benchmarks.py --data_dir path/to/dataset --skip_vis
```

This will skip the visualization step but still generate CSV results and HTML report. 