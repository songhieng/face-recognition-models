import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .model import preprocess_image, extract_features

def get_image_paths(folder, exts=(".jpg", ".jpeg", ".png")):
    """
    Get all image file paths from a given directory.
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def extract_embeddings(model, img_paths):
    """
    Extract embeddings from images using AdaFace.
    Returns an (N, D) numpy array.
    """
    embeddings = []

    for img_path in tqdm(img_paths, desc="Extracting features"):
        img_tensor = preprocess_image(img_path)
        if img_tensor is not None:
            embedding = extract_features(model, img_tensor)
            if embedding is not None:
                embeddings.append(embedding.flatten())  # Flatten to ensure proper stacking

    return np.vstack(embeddings) if embeddings else None  # Ensure a valid NumPy array

def compute_pairwise_metrics(emb1, emb2, threshold=0.4, within=False):
    """
    If within=True, computes all unique pairs within emb1==emb2:
      uses upper-triangular of cosine_similarity.
    Else, computes full cross-similarity between emb1 and emb2.
    Returns (n_pairs, mean_sim, ver_rate).
    """
    # Compute cosine similarity
    sims = np.matmul(emb1, emb2.T)
    
    if within:
        # Take only upper-triangular, remove diagonal
        assert emb1.shape[0] == emb2.shape[0], "Within-case requires equal sets"
        i, j = np.triu_indices(sims.shape[0], k=1)
        sims_flat = sims[i, j]
    else:
        sims_flat = sims.flatten()
        
    if sims_flat.size == 0:
        return 0, float("nan"), float("nan")
        
    mean_sim = sims_flat.mean()
    ver_rate = np.mean(sims_flat >= threshold)
    return sims_flat.size, mean_sim, ver_rate

def benchmark(dataset_dir, model, threshold=0.4):
    """
    Walks each person_ID folder under dataset_dir and prints:
       FF: frontal–frontal
       PP: profile–profile
       FP: frontal–profile
    Returns a list of per-person results.
    """
    results = []
    
    # Get all person directories
    person_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    for person_id in tqdm(person_dirs, desc="Processing individuals"):
        person_dir = os.path.join(dataset_dir, person_id)
        
        fr_dir = os.path.join(person_dir, "frontal")
        pr_dir = os.path.join(person_dir, "profile")
        
        if not (os.path.isdir(fr_dir) and os.path.isdir(pr_dir)):
            print(f"→ Skipping {person_id}: missing 'frontal' or 'profile' folder")
            continue
            
        fr_imgs = get_image_paths(fr_dir)
        pr_imgs = get_image_paths(pr_dir)
        
        if not fr_imgs or not pr_imgs:
            print(f"→ Skipping {person_id}: no images in one of the subfolders")
            continue
            
        # 1) Extract embeddings
        print(f"Processing {person_id}: frontal images")
        emb_fr = extract_embeddings(model, fr_imgs)
        
        print(f"Processing {person_id}: profile images")
        emb_pr = extract_embeddings(model, pr_imgs)
        
        if emb_fr is None or emb_pr is None:
            print(f"→ Skipping {person_id}: could not extract embeddings")
            continue
            
        # 2) Compute metrics
        n_ff, mean_ff, ver_ff = compute_pairwise_metrics(emb_fr, emb_fr, threshold, within=True)
        n_pp, mean_pp, ver_pp = compute_pairwise_metrics(emb_pr, emb_pr, threshold, within=True)
        n_fp, mean_fp, ver_fp = compute_pairwise_metrics(emb_fr, emb_pr, threshold, within=False)
        
        # 3) Print per-person
        print(f"{person_id:>4}  | "
              f"FF pairs={n_ff:3d}, μ={mean_ff:.3f}, acc={ver_ff*100:5.1f}%  | "
              f"PP pairs={n_pp:3d}, μ={mean_pp:.3f}, acc={ver_pp*100:5.1f}%  | "
              f"FP pairs={n_fp:3d}, μ={mean_fp:.3f}, acc={ver_fp*100:5.1f}%")
              
        results.append({
            "person": person_id,
            "ff_pairs": n_ff,    "ff_mean": mean_ff,   "ff_acc": ver_ff,
            "pp_pairs": n_pp,    "pp_mean": mean_pp,   "pp_acc": ver_pp,
            "fp_pairs": n_fp,    "fp_mean": mean_fp,   "fp_acc": ver_fp,
        })
        
    return results

def evaluate_protocol_splits(dataset_dir, protocol_dir, model, threshold=0.4):
    """
    Evaluate the model on the standard CFP protocol splits
    """
    results = {
        'FF': [],
        'FP': []
    }
    
    # Load pair list files
    with open(os.path.join(protocol_dir, 'Pair_list_F.txt'), 'r') as f:
        frontal_list = f.readlines()
    
    with open(os.path.join(protocol_dir, 'Pair_list_P.txt'), 'r') as f:
        profile_list = f.readlines()
    
    # Convert to dictionary for faster lookup
    frontal_dict = {}
    for line in frontal_list:
        parts = line.strip().split()
        if len(parts) == 2:
            index, path = parts
            frontal_dict[index] = path
    
    profile_dict = {}
    for line in profile_list:
        parts = line.strip().split()
        if len(parts) == 2:
            index, path = parts
            profile_dict[index] = path
    
    # Process each split type (FF and FP)
    for split_type in ['FF', 'FP']:
        split_dir = os.path.join(protocol_dir, 'Split', split_type)
        split_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        
        for split_folder in tqdm(split_folders, desc=f"Processing {split_type} splits"):
            split_path = os.path.join(split_dir, split_folder)
            
            # Process same pairs
            same_pairs = []
            with open(os.path.join(split_path, 'same.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        same_pairs.append((parts[0], parts[1]))
            
            # Process different pairs
            diff_pairs = []
            with open(os.path.join(split_path, 'diff.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        diff_pairs.append((parts[0], parts[1]))
            
            # Extract features and compute similarity for all pairs
            y_true = []
            y_scores = []
            
            # Process same pairs
            for idx1, idx2 in tqdm(same_pairs, desc=f"Processing same pairs in {split_folder}"):
                if split_type == 'FF':
                    img1_path = os.path.join(dataset_dir, frontal_dict[idx1])
                    img2_path = os.path.join(dataset_dir, frontal_dict[idx2])
                else:  # FP
                    img1_path = os.path.join(dataset_dir, frontal_dict[idx1])
                    img2_path = os.path.join(dataset_dir, profile_dict[idx2])
                
                img1_tensor = preprocess_image(img1_path)
                img2_tensor = preprocess_image(img2_path)
                
                if img1_tensor is not None and img2_tensor is not None:
                    feat1 = extract_features(model, img1_tensor)
                    feat2 = extract_features(model, img2_tensor)
                    
                    if feat1 is not None and feat2 is not None:
                        similarity = np.dot(feat1.flatten(), feat2.flatten())
                        y_scores.append(similarity)
                        y_true.append(1)  # 1 for same pair
            
            # Process different pairs
            for idx1, idx2 in tqdm(diff_pairs, desc=f"Processing different pairs in {split_folder}"):
                if split_type == 'FF':
                    img1_path = os.path.join(dataset_dir, frontal_dict[idx1])
                    img2_path = os.path.join(dataset_dir, frontal_dict[idx2])
                else:  # FP
                    img1_path = os.path.join(dataset_dir, frontal_dict[idx1])
                    img2_path = os.path.join(dataset_dir, profile_dict[idx2])
                
                img1_tensor = preprocess_image(img1_path)
                img2_tensor = preprocess_image(img2_path)
                
                if img1_tensor is not None and img2_tensor is not None:
                    feat1 = extract_features(model, img1_tensor)
                    feat2 = extract_features(model, img2_tensor)
                    
                    if feat1 is not None and feat2 is not None:
                        similarity = np.dot(feat1.flatten(), feat2.flatten())
                        y_scores.append(similarity)
                        y_true.append(0)  # 0 for different pair
            
            # Calculate metrics for this split
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            
            # Accuracy at threshold
            predictions = (y_scores >= threshold).astype(int)
            accuracy = np.mean(predictions == y_true)
            
            # ROC and AUC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Find EER (Equal Error Rate)
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            
            split_result = {
                'split': split_folder,
                'accuracy': accuracy,
                'auc': roc_auc,
                'eer': eer,
                'threshold': threshold
            }
            
            results[split_type].append(split_result)
            
            print(f"{split_type} Split {split_folder}: Acc={accuracy*100:.2f}%, AUC={roc_auc:.4f}, EER={eer:.4f}")
    
    return results 