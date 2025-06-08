import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.elasticface import load_model, compare_faces

def main():
    parser = argparse.ArgumentParser(description='Visualize face comparison results')
    parser.add_argument('--backbone', type=str, default='pretrain-model/295672backbone.pth', 
                        help='Path to backbone checkpoint')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Similarity threshold for verification')
    parser.add_argument('--output', type=str, default='results/comparison.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.backbone):
        print(f"Error: Backbone file not found at {args.backbone}")
        return
    
    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return
    
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    print(f"Loading model...")
    try:
        model = load_model(args.backbone)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Compare faces
    print(f"Comparing faces...")
    try:
        # Get similarity and match result as a tuple
        similarity, is_match = compare_faces(model, args.image1, args.image2, args.threshold)
        
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
        plt.suptitle(f"Similarity: {similarity:.4f} ({match_text})", 
                     color=color, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save figure
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output}")
        
        # Show plot if running interactively
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing comparison: {e}")

if __name__ == '__main__':
    main() 