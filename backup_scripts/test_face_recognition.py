import argparse
import os
from models.elasticface import load_model, compare_faces

def main():
    parser = argparse.ArgumentParser(description='Test face recognition models')
    parser.add_argument('--model', type=str, default='elasticface', choices=['elasticface'], 
                        help='Face recognition model to use')
    parser.add_argument('--backbone', type=str, default='pretrain-model/295672backbone.pth', 
                        help='Path to backbone checkpoint')
    parser.add_argument('--image1', type=str, default='test/out.jpg',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='test/out1.png',
                        help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Similarity threshold for verification')
    
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
    
    # Load model
    print(f"Loading {args.model} model...")
    try:
        model = load_model(args.backbone)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Compare faces
    print(f"Comparing faces in: {args.image1} and {args.image2}")
    
    try:
        # Get similarity and match result as a tuple
        similarity, is_match = compare_faces(model, args.image1, args.image2, args.threshold)
        
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {is_match}")
        print(f"Threshold used: {args.threshold}")
    except Exception as e:
        print(f"Error comparing faces: {e}")

if __name__ == '__main__':
    main() 