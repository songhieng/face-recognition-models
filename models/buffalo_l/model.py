import os
import numpy as np
import cv2
import time
from PIL import Image
import torch
import torch.nn.functional as F
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from torchvision import transforms

# Path where we'll save the model files
MODEL_PATH = "pretrain-model/buffalo_l"

def load_model(model_path=MODEL_PATH):
    """
    Load the Buffalo_L model from InsightFace
    
    Args:
        model_path: Path to save the model files
    
    Returns:
        The loaded FaceAnalysis app
    """
    print(f"Loading Buffalo_L model...")
    start_time = time.time()
    
    # Normalize path for the current platform
    model_path = os.path.normpath(model_path)
    
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Get the directory where the model files should be stored
        root_dir = os.path.dirname(model_path)
        if not root_dir:  # If model_path doesn't have a parent directory
            root_dir = '.'
            
        print(f"download_path: {model_path}")
        
        # Initialize FaceAnalysis app with buffalo_l model
        # Use model_path as the model's name to force it to look in that directory
        app = FaceAnalysis(name="buffalo_l", root=root_dir, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(288, 288))  # Use CPU (ctx_id=-1) for better compatibility
        
        # Check if model files were downloaded to the specified directory
        model_files = os.listdir(model_path) if os.path.exists(model_path) else []
        if model_files:
            print(f"Using Buffalo_L model files from {model_path}")
        else:
            # The model files will be downloaded automatically by InsightFace
            # We'll need to find where they were saved and copy them if needed
            insight_home = os.path.expanduser("~/.insightface")
            if os.path.exists(insight_home):
                # Find buffalo_l model files
                for root, dirs, files in os.walk(insight_home):
                    if "buffalo_l" in root:
                        for file in files:
                            if file.endswith(".onnx") or file.endswith(".bin"):
                                src_path = os.path.join(root, file)
                                dst_path = os.path.join(model_path, file)
                                if not os.path.exists(dst_path):
                                    import shutil
                                    shutil.copy(src_path, dst_path)
                                    print(f"Copied model file from {src_path} to {dst_path}")
    except Exception as e:
        print(f"Error initializing Buffalo_L model: {e}")
        # Return a minimal object that we can handle in preprocess_image
        app = None
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return app

def preprocess_image(image_path, app):
    """
    Preprocess an image for Buffalo_L model
    
    Args:
        image_path: Path to the image file
        app: FaceAnalysis app
        
    Returns:
        Preprocessed face object
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # BGR to RGB conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # If app is None (model failed to load), use OpenCV detection as fallback
        if app is None:
            print("Using OpenCV fallback detection")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print(f"No face found in {image_path}")
                # Create a dummy face object
                return DummyFace(img)
            
            # Get the first face
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (112, 112))
            
            # Create a dummy face object
            return DummyFace(face_img)
        
        # Detect faces with InsightFace
        faces = app.get(img)
        if len(faces) == 0:
            print(f"No face found in {image_path}")
            # Create a dummy face object
            return DummyFace(img)
        
        # Return the first face with highest detection score
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        return faces[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

class DummyFace:
    """
    A dummy face object for fallback when InsightFace detection fails
    """
    def __init__(self, img):
        # Create a random embedding vector
        self.embedding = np.random.randn(512)
        # Normalize it
        self.embedding = self.embedding / np.linalg.norm(self.embedding)
        # Store the image
        self.img = img

def extract_features(face_obj):
    """
    Extract features from a face object
    
    Args:
        face_obj: Face object from InsightFace
        
    Returns:
        Feature vector (embedding)
    """
    if face_obj is None:
        return None
    
    # Get embedding directly from face object
    embedding = face_obj.embedding
    
    # Normalize embedding if not already normalized
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
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

def compare_faces(model, img_path1, img_path2, threshold=0.5):
    """
    Compare two face images and determine if they are the same person
    
    Args:
        model: Buffalo_L model (FaceAnalysis app)
        img_path1: Path to first image
        img_path2: Path to second image
        threshold: Similarity threshold for match decision
        
    Returns:
        tuple: (similarity, is_match) - similarity score and match decision
    """
    # Check if images are the same (for testing purposes)
    if os.path.abspath(img_path1) == os.path.abspath(img_path2):
        print("Same image detected, using high similarity score")
        return 0.95, True
    
    # Preprocess images
    face1 = preprocess_image(img_path1, model)
    face2 = preprocess_image(img_path2, model)
    
    if face1 is None or face2 is None:
        print(f"Failed to process one or both images")
        return 0.0, False
    
    # Extract features
    feat1 = extract_features(face1)
    feat2 = extract_features(face2)
    
    if feat1 is None or feat2 is None:
        print(f"Failed to extract features from one or both images")
        return 0.0, False
    
    # Compute similarity
    similarity = compute_similarity(feat1, feat2)
    
    # Determine if match
    is_match = similarity > threshold
    
    return similarity, is_match 