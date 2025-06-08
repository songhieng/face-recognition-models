import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

from models.elasticface import load_model as load_elasticface
from models.adaface import load_model as load_adaface
from models.facenet512 import load_model as load_facenet512
from models.buffalo_l import load_model as load_buffalo_l

from models.elasticface import compare_faces as compare_elasticface
from models.adaface import compare_faces as compare_adaface
from models.facenet512 import compare_faces as compare_facenet512
from models.buffalo_l import compare_faces as compare_buffalo_l

def main():
    parser = argparse.ArgumentParser(description='Compare all face recognition models')
    parser.add_argument('--elasticface_path', type=str, default='pretrain-model/295672backbone.pth',
                        help='Path to ElasticFace model')
    parser.add_argument('--adaface_path', type=str, default='pretrain-model/adaface_ir101_webface12m.ckpt',
                        help='Path to AdaFace model')
    parser.add_argument('--facenet512_path', type=str, default='pretrain-model/facenet512_model.h5',
                        help='Path to FaceNet512 model')
    parser.add_argument('--buffalo_l_path', type=str, default='pretrain-model/buffalo_l',
                        help='Path to Buffalo_L model')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to second image')
    parser.add_argument('--output', type=str, default='results/all_models_comparison.png',
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
    
    # Dictionary to store results
    results = {}
    
    # Load and test ElasticFace
    print("\nLoading ElasticFace model...")
    try:
        elasticface_model = load_elasticface(args.elasticface_path)
        print("ElasticFace model loaded successfully")
        
        start_time = time.time()
        elasticface_similarity, elasticface_match = compare_elasticface(
            elasticface_model, args.image1, args.image2, threshold=0.3
        )
        elasticface_time = time.time() - start_time
        
        results['ElasticFace'] = {
            'similarity': elasticface_similarity,
            'match': elasticface_match,
            'time': elasticface_time,
            'threshold': 0.3
        }
        
        print(f"ElasticFace comparison completed in {elasticface_time:.3f} seconds")
        print(f"ElasticFace similarity score: {elasticface_similarity:.4f}")
        print(f"ElasticFace match: {elasticface_match}")
    except Exception as e:
        print(f"Error with ElasticFace model: {e}")
    
    # Load and test AdaFace
    print("\nLoading AdaFace model...")
    try:
        adaface_model = load_adaface(args.adaface_path)
        print("AdaFace model loaded successfully")
        
        start_time = time.time()
        adaface_similarity, adaface_match = compare_adaface(
            adaface_model, args.image1, args.image2, threshold=0.4
        )
        adaface_time = time.time() - start_time
        
        results['AdaFace'] = {
            'similarity': adaface_similarity,
            'match': adaface_match,
            'time': adaface_time,
            'threshold': 0.4
        }
        
        print(f"AdaFace comparison completed in {adaface_time:.3f} seconds")
        print(f"AdaFace similarity score: {adaface_similarity:.4f}")
        print(f"AdaFace match: {adaface_match}")
    except Exception as e:
        print(f"Error with AdaFace model: {e}")
    
    # Load and test FaceNet512
    print("\nLoading FaceNet512 model...")
    try:
        facenet512_model = load_facenet512(args.facenet512_path)
        print("FaceNet512 model loaded successfully")
        
        start_time = time.time()
        facenet512_similarity, facenet512_match = compare_facenet512(
            facenet512_model, args.image1, args.image2, threshold=0.4
        )
        facenet512_time = time.time() - start_time
        
        results['FaceNet512'] = {
            'similarity': facenet512_similarity,
            'match': facenet512_match,
            'time': facenet512_time,
            'threshold': 0.4
        }
        
        print(f"FaceNet512 comparison completed in {facenet512_time:.3f} seconds")
        print(f"FaceNet512 similarity score: {facenet512_similarity:.4f}")
        print(f"FaceNet512 match: {facenet512_match}")
    except Exception as e:
        print(f"Error with FaceNet512 model: {e}")
    
    # Load and test Buffalo_L
    print("\nLoading Buffalo_L model...")
    try:
        # Normalize the Buffalo_L path for the current platform
        buffalo_l_path = os.path.normpath(args.buffalo_l_path)
        buffalo_l_model = load_buffalo_l(buffalo_l_path)
        print("Buffalo_L model loaded successfully")
        
        start_time = time.time()
        buffalo_l_similarity, buffalo_l_match = compare_buffalo_l(
            buffalo_l_model, args.image1, args.image2, threshold=0.5
        )
        buffalo_l_time = time.time() - start_time
        
        results['Buffalo_L'] = {
            'similarity': buffalo_l_similarity,
            'match': buffalo_l_match,
            'time': buffalo_l_time,
            'threshold': 0.5
        }
        
        print(f"Buffalo_L comparison completed in {buffalo_l_time:.3f} seconds")
        print(f"Buffalo_L similarity score: {buffalo_l_similarity:.4f}")
        print(f"Buffalo_L match: {buffalo_l_match}")
    except Exception as e:
        print(f"Error with Buffalo_L model: {e}")
    
    # Create visualization
    if results:
        try:
            # Create a grid for the models and images
            fig = plt.figure(figsize=(12, 10))
            
            # Top row: original images
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            
            # Load images
            img1 = Image.open(args.image1)
            img2 = Image.open(args.image2)
            
            ax1.imshow(np.array(img1))
            ax1.set_title("Image 1")
            ax1.axis('off')
            
            ax2.imshow(np.array(img2))
            ax2.set_title("Image 2")
            ax2.axis('off')
            
            # Bottom area: model results
            ax_results = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)
            ax_results.axis('off')
            
            # Create a table of results
            model_names = []
            similarities = []
            match_results = []
            times = []
            thresholds = []
            
            for model_name, model_results in results.items():
                model_names.append(model_name)
                similarities.append(f"{model_results['similarity']:.4f}")
                match_results.append("MATCH" if model_results['match'] else "NO MATCH")
                times.append(f"{model_results['time']:.3f}s")
                thresholds.append(f"{model_results['threshold']:.2f}")
            
            table_data = [
                similarities,
                match_results,
                thresholds,
                times
            ]
            
            table = ax_results.table(
                cellText=table_data,
                rowLabels=['Similarity', 'Result', 'Threshold', 'Time (s)'],
                colLabels=model_names,
                loc='center',
                cellLoc='center',
                bbox=[0.15, 0.15, 0.7, 0.7]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color the match/no match cells
            for i, model_name in enumerate(model_names):
                if results[model_name]['match']:
                    table[(1, i)].set_facecolor('lightgreen')
                else:
                    table[(1, i)].set_facecolor('lightcoral')
            
            plt.suptitle("Face Recognition Model Comparison", fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.05)
            
            # Save figure
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"\nComparison visualization saved to {args.output}")
            
            # Close plot
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    else:
        print("No results to visualize")

if __name__ == '__main__':
    main() 