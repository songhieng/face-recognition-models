import os
import argparse
import time
import torch
from models.adaface import load_model, compare_faces

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
    parser = argparse.ArgumentParser(description='Test AdaFace face recognition')
    parser.add_argument('--image1', type=str, required=True, help='Path to the first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to the second image')
    parser.add_argument('--threshold', type=float, default=0.4, help='Similarity threshold (default: 0.4)')
    parser.add_argument('--model_path', type=str, default='pretrain-model/adaface_ir101_webface12m.ckpt',
                        help='Path to the pretrained model file (default: pretrain-model/adaface_ir101_webface12m.ckpt)')
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
        print(f"Threshold used: {args.threshold}")
        
        if not os.path.exists(args.model_path) or args.use_demo:
            print("\nNote: These results are based on a demonstration model and are for illustration only.")
            print("For accurate face recognition, please use a properly trained model.")
    except Exception as e:
        print(f"Error comparing faces: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 