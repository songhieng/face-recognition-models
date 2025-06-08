import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models.elasticface import load_model as load_elasticface
from models.elasticface import compare_faces as compare_faces_elasticface
from models.adaface import load_model as load_adaface
from models.adaface import compare_faces as compare_faces_adaface

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
    parser = argparse.ArgumentParser(description='Compare ElasticFace and AdaFace models')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to the first image (default: test/out.jpg)')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to the second image (default: test/out1.png)')
    parser.add_argument('--elasticface_threshold', type=float, default=0.3,
                        help='Similarity threshold for ElasticFace (default: 0.3)')
    parser.add_argument('--adaface_threshold', type=float, default=0.4,
                        help='Similarity threshold for AdaFace (default: 0.4)')
    parser.add_argument('--elasticface_path', type=str, default='pretrain-model/295672backbone.pth',
                        help='Path to the pretrained ElasticFace model file')
    parser.add_argument('--adaface_path', type=str, default='pretrain-model/adaface_ir101_webface12m.ckpt',
                        help='Path to the pretrained AdaFace model file')
    parser.add_argument('--output', type=str, default='results/model_comparison.png',
                        help='Path to save the comparison visualization')
    parser.add_argument('--use_demo', action='store_true', help='Force using the demo AdaFace model')
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
        
    if not os.path.exists(args.elasticface_path):
        print(f"Error: ElasticFace model file {args.elasticface_path} does not exist")
        return
    
    # Check if using demo model for AdaFace
    if args.use_demo:
        args.adaface_path = 'pretrain-model/adaface_demo.ckpt'
        
    # Create demo model if needed or if explicitly requested
    args.adaface_path = create_demo_model_if_needed(args.adaface_path)
        
    # Warn if AdaFace model file doesn't exist
    if not os.path.exists(args.adaface_path):
        print(f"Note: AdaFace model file {args.adaface_path} does not exist. Using random initialization for demo purposes.")
        print("For real-world usage, please download the pretrained AdaFace model file.")
        print("WARNING: AdaFace results with a randomly initialized model will not be accurate.")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load models
    try:
        print(f"Loading ElasticFace model from {args.elasticface_path}...")
        elasticface_model = load_elasticface(args.elasticface_path)
        print("ElasticFace model loaded successfully")
        
        print(f"Loading AdaFace model...")
        adaface_model = load_adaface(args.adaface_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Compare faces with ElasticFace
    try:
        print(f"\nComparing faces with ElasticFace...")
        elastic_start_time = time.time()
        elastic_similarity, elastic_match = compare_faces_elasticface(
            elasticface_model, args.image1, args.image2, args.elasticface_threshold
        )
        elastic_end_time = time.time()
        elastic_time = elastic_end_time - elastic_start_time
        
        print(f"ElasticFace comparison completed in {elastic_time:.3f} seconds")
        print(f"ElasticFace similarity score: {elastic_similarity:.4f}")
        print(f"ElasticFace match: {elastic_match}")
    except Exception as e:
        print(f"Error comparing faces with ElasticFace: {e}")
        return
    
    # Compare faces with AdaFace
    try:
        print(f"\nComparing faces with AdaFace...")
        ada_start_time = time.time()
        ada_similarity, ada_match = compare_faces_adaface(
            adaface_model, args.image1, args.image2, args.adaface_threshold
        )
        ada_end_time = time.time()
        ada_time = ada_end_time - ada_start_time
        
        print(f"AdaFace comparison completed in {ada_time:.3f} seconds")
        print(f"AdaFace similarity score: {ada_similarity:.4f}")
        print(f"AdaFace match: {ada_match}")
    except Exception as e:
        print(f"Error comparing faces with AdaFace: {e}")
        return
    
    # Visualize comparison
    try:
        # Load images
        img1 = Image.open(args.image1)
        img2 = Image.open(args.image2)
        
        # Create figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Images and ElasticFace result
        axs[0, 0].imshow(np.array(img1))
        axs[0, 0].set_title("Image 1")
        axs[0, 0].axis("off")
        
        axs[0, 1].imshow(np.array(img2))
        axs[0, 1].set_title("Image 2")
        axs[0, 1].axis("off")
        
        # ElasticFace visualization
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        gradient = np.repeat(gradient, 50, axis=0)
        
        cmap = 'RdYlGn' if elastic_match else 'RdYlGn_r'
        axs[0, 2].imshow(gradient, cmap=cmap, aspect='auto')
        
        axs[0, 2].text(50, 20, f"ElasticFace", 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        axs[0, 2].text(50, 30, f"Similarity: {elastic_similarity:.4f}", 
                     ha='center', va='center', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        match_text = "MATCH" if elastic_match else "NO MATCH"
        match_color = 'green' if elastic_match else 'red'
        axs[0, 2].text(50, 40, match_text, 
                     ha='center', va='center', fontsize=14, fontweight='bold', color=match_color,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        axs[0, 2].axvline(x=args.elasticface_threshold * 100, color='black', linestyle='--')
        axs[0, 2].text(args.elasticface_threshold * 100 + 5, 10, f"Threshold: {args.elasticface_threshold}", 
                     va='center', fontsize=10, rotation=90)
        
        axs[0, 2].set_title(f"ElasticFace Result (Time: {elastic_time:.3f}s)")
        axs[0, 2].axis("off")
        
        # Row 2: Empty, Empty, AdaFace result
        axs[1, 0].axis("off")
        axs[1, 1].axis("off")
        
        # AdaFace visualization
        cmap = 'RdYlGn' if ada_match else 'RdYlGn_r'
        axs[1, 2].imshow(gradient, cmap=cmap, aspect='auto')
        
        axs[1, 2].text(50, 20, f"AdaFace", 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        axs[1, 2].text(50, 30, f"Similarity: {ada_similarity:.4f}", 
                     ha='center', va='center', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        match_text = "MATCH" if ada_match else "NO MATCH"
        match_color = 'green' if ada_match else 'red'
        axs[1, 2].text(50, 40, match_text, 
                     ha='center', va='center', fontsize=14, fontweight='bold', color=match_color,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        axs[1, 2].axvline(x=args.adaface_threshold * 100, color='black', linestyle='--')
        axs[1, 2].text(args.adaface_threshold * 100 + 5, 10, f"Threshold: {args.adaface_threshold}", 
                     va='center', fontsize=10, rotation=90)
        
        # Add a note if using demo model for AdaFace
        if not os.path.exists(args.adaface_path) or args.use_demo:
            axs[1, 2].text(50, 5, "Demo model - for illustration only", 
                         ha='center', va='center', fontsize=8, style='italic', color='red',
                         bbox=dict(facecolor='white', alpha=0.5))
        
        axs[1, 2].set_title(f"AdaFace Result (Time: {ada_time:.3f}s)")
        axs[1, 2].axis("off")
        
        # Model comparison summary
        # Add a box with comparison summary in the empty area
        comparison_text = (
            f"Model Comparison Summary:\n\n"
            f"ElasticFace:\n"
            f"  - Similarity: {elastic_similarity:.4f}\n"
            f"  - Match: {elastic_match}\n"
            f"  - Processing Time: {elastic_time:.3f}s\n\n"
            f"AdaFace:\n"
            f"  - Similarity: {ada_similarity:.4f}\n"
            f"  - Match: {ada_match}\n"
            f"  - Processing Time: {ada_time:.3f}s\n\n"
            f"Performance Difference:\n"
            f"  - Similarity: {ada_similarity - elastic_similarity:+.4f}\n"
            f"  - Speed: {elastic_time - ada_time:+.3f}s"
        )
        
        # Add a note about demo model if applicable
        if not os.path.exists(args.adaface_path) or args.use_demo:
            comparison_text += "\n\nNote: AdaFace using demo model for illustration"
        
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
        axs[1, 0].text(0.5, 0.5, comparison_text, transform=axs[1, 0].transAxes,
                      fontsize=12, va='center', ha='center', bbox=props)
        
        # Add a centered title
        plt.suptitle(f"Face Recognition Model Comparison", fontsize=16, y=0.98)
        
        # Add a horizontal bar chart in the bottom-middle to compare scores
        bar_width = 0.35
        x = np.array([0, 1])
        similarities = np.array([elastic_similarity, ada_similarity])
        
        bars = axs[1, 1].bar(x, similarities, bar_width, color=['blue', 'orange'])
        
        # Add threshold lines
        axs[1, 1].axhline(y=args.elasticface_threshold, color='blue', linestyle='--', alpha=0.5)
        axs[1, 1].text(0, args.elasticface_threshold + 0.02, f"ElasticFace\nThreshold", 
                     ha='center', va='bottom', color='blue', fontsize=8)
        
        axs[1, 1].axhline(y=args.adaface_threshold, color='orange', linestyle='--', alpha=0.5)
        axs[1, 1].text(1, args.adaface_threshold + 0.02, f"AdaFace\nThreshold", 
                     ha='center', va='bottom', color='orange', fontsize=8)
        
        # Add bar labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                         f'{height:.4f}', ha='center', va='bottom')
        
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(['ElasticFace', 'AdaFace'])
        axs[1, 1].set_ylim(0, max(1.0, max(similarities) + 0.1))
        axs[1, 1].set_title("Similarity Score Comparison")
        
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to {args.output}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 