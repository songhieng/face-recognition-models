import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import time
from torchvision import transforms
from deepface import DeepFace
from deepface.commons import functions

# Path where we'll save the model weights
MODEL_PATH = "pretrain-model/facenet512_model.h5"

def load_model(model_path=MODEL_PATH):
    """
    Load the FaceNet512 model from DeepFace
    
    Args:
        model_path: Path to save the model weights
    
    Returns:
        The loaded model
    """
    print(f"Loading FaceNet512 model...")
    start_time = time.time()
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load model with DeepFace
    model = DeepFace.build_model("Facenet512")
    
    # Save the model weights if they don't exist yet
    if not os.path.exists(model_path):
        # Get the original model path from DeepFace
        original_model_path = functions.get_deepface_home() + "/weights/facenet512_weights.h5"
        
        # Use shutil to copy the file if it exists
        if os.path.exists(original_model_path):
            import shutil
            shutil.copy(original_model_path, model_path)
            print(f"Model weights saved to {model_path}")
        else:
            print(f"Original model weights not found at {original_model_path}")
    else:
        print(f"Using existing model weights from {model_path}")
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model

def preprocess_image(image_path):
    """
    Preprocess an image for FaceNet512
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image ready for the model
    """
    try:
        # Use simpler approach with direct OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
            
        # Convert to RGB (from BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face using OpenCV's Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print(f"No face found in {image_path}")
            # Use the whole image as fallback
            face_img = cv2.resize(img, (160, 160))
        else:
            # Get the first face
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
        
        # Normalize pixel values
        face_img = face_img.astype('float32')
        face_img /= 255.0
        
        # Expand dimensions for model input
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_features(model, img):
    """
    Extract features from a preprocessed face image
    
    Args:
        model: FaceNet512 model
        img: Preprocessed face image
        
    Returns:
        Feature vector (embedding)
    """
    if img is None:
        return None
    
    # Get embedding from the model
    embedding = model.predict(img)[0]
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def compute_similarity(feat1, feat2):
    """
    Compute cosine similarity between two feature vectors
    
    Args:
        feat1: First feature vector
        feat2: Second feature vector
        
    Returns:
        similarity: Cosine similarity score
    """
    if feat1 is None or feat2 is None:
        return 0.0
    
    # Compute cosine similarity
    similarity = np.dot(feat1, feat2)
    
    return float(similarity)

def compare_faces(model, img_path1, img_path2, threshold=0.4):
    """
    Compare two face images and determine if they are the same person
    
    Args:
        model: FaceNet512 model
        img_path1: Path to first image
        img_path2: Path to second image
        threshold: Similarity threshold for match decision
        
    Returns:
        tuple: (similarity, is_match) - similarity score and match decision
    """
    # Preprocess images
    img1 = preprocess_image(img_path1)
    img2 = preprocess_image(img_path2)
    
    if img1 is None or img2 is None:
        print(f"Failed to process one or both images")
        return 0.0, False
    
    # Extract features
    feat1 = extract_features(model, img1)
    feat2 = extract_features(model, img2)
    
    if feat1 is None or feat2 is None:
        print(f"Failed to extract features from one or both images")
        return 0.0, False
    
    # Compute similarity
    similarity = compute_similarity(feat1, feat2)
    
    # Determine if match
    is_match = similarity > threshold
    
    return similarity, is_match 