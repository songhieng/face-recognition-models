import os
import sys
import time
import base64
import requests
from PIL import Image
from io import BytesIO

# Add current directory to path
sys.path.append('.')

# Import AdaFace functions
from models.adaface import load_model, compare_faces

def save_images():
    """Save the attached images to disk"""
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # Get paths to the images
    image1_path = 'temp/1.jpg'
    image2_path = 'temp/2.jpg'
    
    # These images come from user attachments
    # Try to copy images that were already attached by the user
    try:
        from shutil import copyfile
        # Assuming the images are named 1.jpg and 2.jpg in the attachment
        if os.path.exists('1.jpg'):
            copyfile('1.jpg', image1_path)
        
        if os.path.exists('2.jpg'):
            copyfile('2.jpg', image2_path)
    except Exception as e:
        print(f"Error saving attached images: {e}")
    
    return image1_path, image2_path

def main():
    # Fixed paths to the images
    image1 = "temp/1.jpg"
    image2 = "temp/2.jpg"
    threshold = 0.4  # Default similarity threshold
    model_path = "pretrain-model/adaface_ir101_webface12m.ckpt"
    
    # Check if files exist
    if not os.path.exists(image1):
        print(f"Error: Image file {image1} does not exist")
        return
    
    if not os.path.exists(image2):
        print(f"Error: Image file {image2} does not exist")
        return
    
    # Warn if model file doesn't exist
    if not os.path.exists(model_path):
        print(f"Note: Model file {model_path} does not exist. Using random initialization for demo purposes.")
        print("For real-world usage, please download the pretrained model file.")
    
    # Load model
    try:
        print(f"Loading AdaFace model...")
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare faces
    try:
        print(f"Comparing faces in {image1} and {image2}...")
        start_time = time.time()
        similarity, match = compare_faces(model, image1, image2, threshold)
        end_time = time.time()
        
        print(f"Comparison completed in {end_time - start_time:.3f} seconds")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Match: {match}")
        print(f"Threshold used: {threshold}")
    except Exception as e:
        print(f"Error comparing faces: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 