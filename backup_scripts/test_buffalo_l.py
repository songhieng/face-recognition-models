import argparse
import os
import time
from models.buffalo_l import load_model, compare_faces

def main():
    parser = argparse.ArgumentParser(description='Test Buffalo_L face recognition model')
    parser.add_argument('--model_path', type=str, default='pretrain-model/buffalo_l',
                        help='Path to model directory')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for verification')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return
    
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return
    
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
        print(f"Threshold used: {args.threshold}")
    except Exception as e:
        print(f"Error comparing faces: {e}")

if __name__ == '__main__':
    main() 