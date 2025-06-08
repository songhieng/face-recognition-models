import os
import time
import argparse
import torch

# Import model loading functions
from models.elasticface import load_model as load_elasticface
from models.adaface import load_model as load_adaface
from models.facenet512 import load_model as load_facenet512
from models.buffalo_l import load_model as load_buffalo_l

# Import comparison functions
from models.elasticface import compare_faces as compare_elasticface
from models.adaface import compare_faces as compare_adaface
from models.facenet512 import compare_faces as compare_facenet512
from models.buffalo_l import compare_faces as compare_buffalo_l

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test all face recognition models on sample images")
    parser.add_argument("--image1", type=str, default="test/out.jpg", help="Path to first image")
    parser.add_argument("--image2", type=str, default="test/out1.png", help="Path to second image")
    parser.add_argument("--use_demo", action="store_true", help="Use demo models for ElasticFace and AdaFace")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()
    
    # Check CUDA availability
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"\nUsing device: {device}")
    if use_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Check if images exist
    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return
    
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return
    
    print("\n" + "="*60)
    print("TESTING ALL FACE RECOGNITION MODELS")
    print("="*60)
    
    # Test ElasticFace
    print("\n" + "-"*60)
    print("Testing ElasticFace model")
    print("-"*60)
    
    elasticface_path = "pretrain-model/elasticface_demo.ckpt" if args.use_demo else "pretrain-model/295672backbone.pth"
    try:
        print(f"Loading ElasticFace model from {elasticface_path}...")
        model = load_elasticface(elasticface_path, device=device)
        print("Model loaded successfully")
        
        start_time = time.time()
        similarity, is_match = compare_elasticface(model, args.image1, args.image2, threshold=0.3, device=device)
        elapsed_time = time.time() - start_time
        
        print(f"Comparison completed in {elapsed_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {'Yes' if is_match else 'No'}")
        print(f"Threshold: 0.3")
    except Exception as e:
        print(f"Error testing ElasticFace model: {e}")
    
    # Test AdaFace
    print("\n" + "-"*60)
    print("Testing AdaFace model")
    print("-"*60)
    
    adaface_path = "pretrain-model/adaface_demo.ckpt" if args.use_demo else "pretrain-model/adaface_ir101_webface12m.ckpt"
    try:
        print(f"Loading AdaFace model from {adaface_path}...")
        model = load_adaface(adaface_path, device=device)
        print("Model loaded successfully")
        
        start_time = time.time()
        similarity, is_match = compare_adaface(model, args.image1, args.image2, threshold=0.4, device=device)
        elapsed_time = time.time() - start_time
        
        print(f"Comparison completed in {elapsed_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {'Yes' if is_match else 'No'}")
        print(f"Threshold: 0.4")
    except Exception as e:
        print(f"Error testing AdaFace model: {e}")
    
    # Test FaceNet512
    print("\n" + "-"*60)
    print("Testing FaceNet512 model")
    print("-"*60)
    
    facenet512_path = "pretrain-model/facenet512_model.h5"
    try:
        print(f"Loading FaceNet512 model from {facenet512_path}...")
        # FaceNet512 handles CUDA internally through TensorFlow
        model = load_facenet512(facenet512_path)
        print("Model loaded successfully")
        
        start_time = time.time()
        similarity, is_match = compare_facenet512(model, args.image1, args.image2, threshold=0.4)
        elapsed_time = time.time() - start_time
        
        print(f"Comparison completed in {elapsed_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {'Yes' if is_match else 'No'}")
        print(f"Threshold: 0.4")
    except Exception as e:
        print(f"Error testing FaceNet512 model: {e}")
    
    # Test Buffalo_L
    print("\n" + "-"*60)
    print("Testing Buffalo_L model")
    print("-"*60)
    
    buffalo_l_path = os.path.normpath("pretrain-model/buffalo_l")
    try:
        print(f"Loading Buffalo_L model from {buffalo_l_path}...")
        # Buffalo_L handles GPU usage internally through ONNXRuntime
        model = load_buffalo_l(buffalo_l_path)
        print("Model loaded successfully")
        
        start_time = time.time()
        similarity, is_match = compare_buffalo_l(model, args.image1, args.image2, threshold=0.5)
        elapsed_time = time.time() - start_time
        
        print(f"Comparison completed in {elapsed_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {'Yes' if is_match else 'No'}")
        print(f"Threshold: 0.5")
    except Exception as e:
        print(f"Error testing Buffalo_L model: {e}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main() 