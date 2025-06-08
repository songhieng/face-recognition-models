import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from .backbone import iresnet50

class ElasticFace(nn.Module):
    def __init__(self, backbone, feat_dim=512, s=64.0, m=0.5):
        super(ElasticFace, self).__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.s = s
        self.m = m
        
    def forward(self, x, label=None):
        x = self.backbone(x)
        return x

def load_model(backbone_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create backbone
    backbone = iresnet50(dropout=0, fp16=False, num_features=512)
    
    # Load backbone checkpoint
    checkpoint = torch.load(backbone_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        backbone.load_state_dict(checkpoint, strict=False)
    
    # Create ElasticFace model
    model = ElasticFace(backbone=backbone)
    model.eval()
    model = model.to(device)
    
    return model

def preprocess_image(image_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        return img_tensor
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_features(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor)
        features = F.normalize(features, p=2, dim=1)  # L2 normalization
    return features

def compute_similarity(feat1, feat2):
    similarity = torch.sum(feat1 * feat2).item()
    return similarity

def compare_faces(model, img_path1, img_path2, threshold=0.5, device=None):
    """
    Compare two face images and determine if they are the same person.
    
    Args:
        model: ElasticFace model
        img_path1: Path to first image
        img_path2: Path to second image
        threshold: Similarity threshold for match decision
        device: Torch device (cuda or cpu)
        
    Returns:
        tuple: (similarity, is_match) - a tuple containing the similarity score and a boolean indicating if the faces match
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess images
    img1_tensor = preprocess_image(img_path1, device)
    img2_tensor = preprocess_image(img_path2, device)
    
    if img1_tensor is None or img2_tensor is None:
        raise ValueError("Failed to process one or both images")
    
    # Extract features
    feat1 = extract_features(model, img1_tensor)
    feat2 = extract_features(model, img2_tensor)
    
    if feat1 is None or feat2 is None:
        raise ValueError("Failed to extract features from one or both images")
    
    # Compute similarity
    similarity = compute_similarity(feat1, feat2)
    
    # Determine if match
    is_match = similarity > threshold
    
    # Return as tuple
    return float(similarity), bool(is_match) 