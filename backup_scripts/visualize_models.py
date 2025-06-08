import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import json

from test_models import test_model

def visualize_comparison(results, image1_path, image2_path, output_path):
    """Create a visualization of face comparison results from multiple models"""
    if not results:
        print("No results to visualize")
        return
    
    # Load images
    try:
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Determine grid size based on number of models
    num_models = len(results)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Display original images at the top
    ax1 = plt.subplot2grid((3, max(2, num_models)), (0, 0))
    ax2 = plt.subplot2grid((3, max(2, num_models)), (0, 1))
    
    ax1.imshow(np.array(img1))
    ax1.set_title("Image 1")
    ax1.axis('off')
    
    ax2.imshow(np.array(img2))
    ax2.set_title("Image 2")
    ax2.axis('off')
    
    # Create a table for model results
    model_names = []
    similarities = []
    match_results = []
    times = []
    thresholds = []
    
    for result in results:
        model_names.append(result['model'])
        similarities.append(f"{result['similarity']:.4f}")
        match_results.append("MATCH" if result['match'] else "NO MATCH")
        times.append(f"{result['time']:.3f}s")
        thresholds.append(f"{result['threshold']:.2f}")
    
    # Display table of results
    ax_table = plt.subplot2grid((3, max(2, num_models)), (1, 0), colspan=max(2, num_models), rowspan=2)
    ax_table.axis('off')
    
    table_data = [
        similarities,
        match_results,
        thresholds,
        times
    ]
    
    table = ax_table.table(
        cellText=table_data,
        rowLabels=['Similarity', 'Result', 'Threshold', 'Time (s)'],
        colLabels=model_names,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the match/no match cells
    for i, result in enumerate(results):
        if result['match']:
            table[(1, i)].set_facecolor('lightgreen')
        else:
            table[(1, i)].set_facecolor('lightcoral')
    
    plt.suptitle("Face Recognition Model Comparison", fontsize=16, fontweight='bold')
    
    # Add a note about which models were used
    models_used = ", ".join(model_names)
    plt.figtext(0.5, 0.02, f"Models: {models_used}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Close plot
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize face recognition model comparisons')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated list of models to test (elasticface,adaface,facenet512,buffalo_l) or "all"')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to second image')
    parser.add_argument('--use_demo', action='store_true',
                        help='Use demo model for ElasticFace and AdaFace')
    parser.add_argument('--output', type=str, default='results/model_comparison.png',
                        help='Path to save visualization')
    parser.add_argument('--results_json', type=str, default=None,
                        help='Path to load results from (instead of running tests)')
    
    args = parser.parse_args()
    
    # Check if input files exist if we're not loading from JSON
    if not args.results_json:
        if not os.path.exists(args.image1):
            print(f"Error: Image 1 not found at {args.image1}")
            return
        
        if not os.path.exists(args.image2):
            print(f"Error: Image 2 not found at {args.image2}")
            return
    
    # Get results
    results = []
    
    if args.results_json:
        # Load results from JSON file
        try:
            with open(args.results_json, 'r') as f:
                results = json.load(f)
            print(f"Loaded results from {args.results_json}")
        except Exception as e:
            print(f"Error loading results from {args.results_json}: {e}")
            return
    else:
        # Run tests to get results
        # Set model paths and thresholds
        models = {
            'elasticface': {
                'path': 'pretrain-model/295672backbone.pth',
                'threshold': 0.3
            },
            'adaface': {
                'path': 'pretrain-model/adaface_ir101_webface12m.ckpt',
                'threshold': 0.4
            },
            'facenet512': {
                'path': 'pretrain-model/facenet512_model.h5',
                'threshold': 0.4
            },
            'buffalo_l': {
                'path': 'pretrain-model/buffalo_l',
                'threshold': 0.5
            }
        }
        
        # Determine which models to test
        if args.models.lower() == 'all':
            models_to_test = list(models.keys())
        else:
            models_to_test = [m.strip() for m in args.models.split(',')]
            # Filter out invalid models
            models_to_test = [m for m in models_to_test if m in models]
        
        # Test each model
        for model_name in models_to_test:
            if model_name in models:
                result = test_model(
                    model_name, 
                    models[model_name]['path'], 
                    args.image1, 
                    args.image2, 
                    models[model_name]['threshold'],
                    args.use_demo
                )
                if result:
                    results.append(result)
    
    # Create visualization
    if results:
        visualize_comparison(results, args.image1, args.image2, args.output)
    else:
        print("No results to visualize")

if __name__ == '__main__':
    main() 