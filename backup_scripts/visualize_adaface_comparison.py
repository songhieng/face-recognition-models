import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import torch

from models.adaface import load_model, compare_faces, preprocess_image, extract_features

def create_demo_model_if_needed(model_path):
    """Create a demo model if the specified model doesn't exist"""
    if not os.path.exists(model_path):
        try:
            from create_demo_model import create_adaface_demo_model
            demo_path = create_adaface_demo_model()
            print(f"Created demo model at {demo_path}")
            return demo_path
        except ImportError:
            print("Note: create_demo_model.py not found, using random initialization")
            return model_path
    return model_path

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize AdaFace face comparison')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to the first image (default: test/out.jpg)')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to the second image (default: test/out1.png)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Similarity threshold (default: 0.4)')
    parser.add_argument('--model_path', type=str, default='pretrain-model/adaface_ir101_webface12m.ckpt',
                        help='Path to the pretrained model file (default: pretrain-model/adaface_ir101_webface12m.ckpt)')
    parser.add_argument('--output', type=str, default='results/adaface_comparison.png',
                        help='Path to save the visualization (default: results/adaface_comparison.png)')
    parser.add_argument('--use_demo', action='store_true', help='Force using the demo model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image1):
        print(f"Error: Image file {args.image1} does not exist")
        return
    
    if not os.path.exists(args.image2):
        print(f"Error: Image file {args.image2} does not exist")
        return
    
    # Check if using demo model
    if args.use_demo:
        args.model_path = 'pretrain-model/adaface_demo.ckpt'
        
    # Create demo model if needed or if explicitly requested
    args.model_path = create_demo_model_if_needed(args.model_path)
        
    # Warn if model file doesn't exist
    if not os.path.exists(args.model_path):
        print(f"Note: Model file {args.model_path} does not exist. Using random initialization for demo purposes.")
        print("For real-world usage, please download the pretrained model file.")
        print("WARNING: Visualization results with a randomly initialized model will not be accurate.")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    try:
        print(f"Loading AdaFace model...")
        model = load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Compare faces
    try:
        print(f"Comparing faces in {args.image1} and {args.image2}...")
        start_time = time.time()
        similarity, match = compare_faces(model, args.image1, args.image2, args.threshold)
        end_time = time.time()
        
        print(f"Comparison completed in {end_time - start_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {match}")
        
        # Load and display images
        img1 = Image.open(args.image1)
        img2 = Image.open(args.image2)
        
        # Visualize
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display images
        ax[0].imshow(np.array(img1))
        ax[0].set_title("Image 1")
        ax[0].axis("off")
        
        ax[1].imshow(np.array(img2))
        ax[1].set_title("Image 2")
        ax[1].axis("off")
        
        # Display similarity visualization
        # Create a gradient background
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        gradient = np.repeat(gradient, 50, axis=0)
        
        # Show gradient with color based on match
        cmap = 'RdYlGn' if match else 'RdYlGn_r'
        ax[2].imshow(gradient, cmap=cmap, aspect='auto')
        
        # Add score text
        ax[2].text(50, 25, f"Similarity: {similarity:.4f}", 
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Add match/no match text
        match_text = "MATCH" if match else "NO MATCH"
        match_color = 'green' if match else 'red'
        ax[2].text(50, 40, match_text, 
                 ha='center', va='center', fontsize=14, fontweight='bold', color=match_color,
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Add threshold line
        ax[2].axvline(x=args.threshold * 100, color='black', linestyle='--')
        ax[2].text(args.threshold * 100 + 5, 10, f"Threshold: {args.threshold}", 
                 va='center', fontsize=10, rotation=90)
        
        # Add a note if using demo model
        if not os.path.exists(args.model_path) or args.use_demo:
            fig.text(0.5, 0.01, "Note: Using demonstration model - results are for illustration only", 
                    ha='center', fontsize=10, style='italic', color='red')
        
        ax[2].set_title("Similarity Score")
        ax[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {args.output}")
        
    except Exception as e:
        print(f"Error visualizing comparison: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 