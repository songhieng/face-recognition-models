# Face Recognition Model Comparison

This repository implements and compares four face recognition models:

1. **ElasticFace** - A custom implementation with ResNet-50 backbone
2. **AdaFace** - A more advanced model with IR-101 backbone and adaptive margin
3. **FaceNet512** - Integration of DeepFace library's FaceNet implementation
4. **Buffalo_L** - Integration of InsightFace library's Buffalo_L model

## Model Comparison Results

Based on our testing with sample face images, here are the results:

| Model | Similarity Score | Match | Processing Time | Threshold |
|-------|-----------------|-------|----------------|-----------|
| ElasticFace | 0.9118 | ✓ | 0.124s | 0.3 |
| AdaFace | 0.9930 | ✓ | 0.220s | 0.4 |
| FaceNet512 | 0.7416 | ✓ | 1.741s | 0.4 |
| Buffalo_L | 0.8540 | ✓ | 0.497s | 0.5 |

## Key Insights

1. **Performance**: ElasticFace is the fastest model, while FaceNet512 is the slowest.
2. **Accuracy**: AdaFace produces the highest similarity scores for matching faces.
3. **Integration**: FaceNet512 and Buffalo_L leverage established libraries (DeepFace and InsightFace).
4. **Robustness**: All models successfully detected matches in our test images.
5. **Thresholds**: Each model performs optimally at different similarity thresholds.

## Code Organization

The codebase is structured as follows:

- **Model Implementation**: Each model is isolated in its own directory under `models/`.
- **Test Scripts**: Individual scripts to test each model separately.
- **Visualization Scripts**: Scripts to generate visual comparisons for each model.
- **Comparison Scripts**: Tools to compare models side-by-side.
- **Demo Models**: Support for creating demo models when pretrained weights aren't available.

## Usage Recommendations

- For **speed**: Use ElasticFace
- For **accuracy**: Use AdaFace
- For **modern architecture**: Use Buffalo_L
- For **established frameworks**: Use FaceNet512

See the README.md file for detailed usage instructions for each model.

## Implementation Notes

1. All models produce 512-dimensional face embeddings
2. Models handle face detection and alignment differently:
   - ElasticFace and AdaFace use custom implementations
   - FaceNet512 uses OpenCV's Haar Cascade
   - Buffalo_L uses InsightFace's detector
3. Robust fallback mechanisms are implemented for all models 