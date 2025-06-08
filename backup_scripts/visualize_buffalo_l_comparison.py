import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from models.buffalo_l import load_model, compare_faces

def main():
    parser = argparse.ArgumentParser(description='Visualize Buffalo_L face comparison results')
    parser.add_argument('--model_path', type=str, default='pretrain-model/buffalo_l',
                        help='Path to model directory')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for verification')
    parser.add_argument('--output', type=str, default='results/buffalo_l_comparison.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return
    
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    print(f"Loading Buffalo_L model...")
    try:
        model = load_model(args.model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Compare faces
    print(f"Comparing faces in {args.image1} and {args.image2}...")
    
    try:
        start_time = time.time()
        similarity, is_match = compare_faces(model, args.image1, args.image2, args.threshold)
        elapsed_time = time.time() - start_time
        
        print(f"Comparison completed in {elapsed_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {is_match}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Load and display images
        img1 = Image.open(args.image1)
        img2 = Image.open(args.image2)
        
        ax1.imshow(np.array(img1))
        ax1.set_title("Image 1")
        ax1.axis('off')
        
        ax2.imshow(np.array(img2))
        ax2.set_title("Image 2")
        ax2.axis('off')
        
        # Add result as suptitle
        match_text = "MATCH" if is_match else "NO MATCH"
        color = "green" if is_match else "red"
        plt.suptitle(f"Buffalo_L - Similarity: {similarity:.4f} ({match_text})", 
                     color=color, fontsize=16, fontweight='bold')
        
        # Add processing time information
        plt.figtext(0.5, 0.01, f"Processing time: {elapsed_time:.3f} seconds", 
                   ha="center", fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.05)
        
        # Save figure
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output}")
        
        # Close plot
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing comparison: {e}")

if __name__ == '__main__':
    main() 