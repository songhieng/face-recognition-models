import os
import argparse
import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# Import models
from models.elasticface import load_model as load_elasticface
from models.adaface import load_model as load_adaface
from models.facenet512 import load_model as load_facenet512
from models.buffalo_l import load_model as load_buffalo_l

def get_image_paths(folder, exts=(".jpg", ".jpeg", ".png")):
    """Get all image file paths from a given directory."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def l2_normalize(x):
    """Normalize vectors to unit length."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def extract_embeddings(model, model_name, img_paths, device):
    """Extract embeddings from images using the specified model."""
    embeddings = []
    
    for img_path in tqdm(img_paths, desc=f"Extracting {model_name} embeddings"):
        try:
            # Load and preprocess image
            if model_name == 'elasticface':
                from models.elasticface.model import preprocess_image
                img = preprocess_image(img_path, device)
                if img is not None:
                    with torch.no_grad():
                        embedding = model(img).cpu().numpy()[0]
                    embeddings.append(embedding)
            
            elif model_name == 'adaface':
                from models.adaface.model import preprocess_image
                img = preprocess_image(img_path, device)
                if img is not None:
                    with torch.no_grad():
                        embedding = model(img).cpu().numpy()[0]
                    embeddings.append(embedding)
            
            elif model_name == 'facenet512':
                from models.facenet512.model import preprocess_image
                img = preprocess_image(img_path)
                if img is not None:
                    embedding = model.predict(img)[0]
                    embeddings.append(embedding)
            
            elif model_name == 'buffalo_l':
                from models.buffalo_l.model import preprocess_image
                img = preprocess_image(img_path)
                if img is not None:
                    embedding = model.get_embedding(img)[0]
                    embeddings.append(embedding)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return np.vstack(embeddings) if embeddings else None

def compute_pairwise_metrics(emb1, emb2, threshold=0.5, within=False):
    """
    Compute similarity metrics between embeddings.
    If within=True, computes all unique pairs within emb1==emb2 (uses upper-triangular).
    Else, computes full cross-similarity between emb1 and emb2.
    """
    sims = cosine_similarity(emb1, emb2)
    if within:
        # take only upper-triangular, remove diagonal
        assert emb1.shape[0] == emb2.shape[0], "Within-case requires equal sets"
        i, j = np.triu_indices(sims.shape[0], k=1)
        sims_flat = sims[i, j]
    else:
        sims_flat = sims.flatten()
    
    if sims_flat.size == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    
    mean_sim = sims_flat.mean()
    std_sim = sims_flat.std()
    min_sim = sims_flat.min()
    max_sim = sims_flat.max()
    ver_rate = np.mean(sims_flat >= threshold)
    
    return sims_flat.size, mean_sim, std_sim, min_sim, max_sim, ver_rate

def calculate_verification_metrics(same_person_sims, diff_person_sims, thresholds=None):
    """Calculate verification metrics (FAR, FRR, EER) at different thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    # Initialize results
    results = []
    
    # Labels for same and different person pairs
    y_true = np.concatenate([np.ones(len(same_person_sims)), np.zeros(len(diff_person_sims))])
    # Scores for all pairs
    y_scores = np.concatenate([same_person_sims, diff_person_sims])
    
    # Calculate ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    eer_threshold = roc_thresholds[eer_threshold_idx]
    
    # Calculate Precision-Recall curve and AP
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    # For each threshold, calculate metrics
    for threshold in thresholds:
        # True/False Positives/Negatives
        tp = np.sum((y_scores >= threshold) & (y_true == 1))
        fp = np.sum((y_scores >= threshold) & (y_true == 0))
        tn = np.sum((y_scores < threshold) & (y_true == 0))
        fn = np.sum((y_scores < threshold) & (y_true == 1))
        
        # Metrics
        far = fp / (fp + tn) if (fp + tn) > 0 else float('nan')  # False Accept Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')  # False Reject Rate
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else float('nan')
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else float('nan')
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'far': far,
            'frr': frr,
            'precision': precision_val,
            'recall': recall_val,
            'f1': f1
        })
    
    return pd.DataFrame(results), roc_auc, eer, eer_threshold, fpr, tpr, precision, recall

def benchmark_model(model, model_name, dataset_dir, threshold, device):
    """
    Benchmark a single model on the dataset.
    Walks each person_ID folder under dataset_dir and computes various metrics.
    """
    results = []
    all_same_sims = []
    all_diff_sims = []
    person_embeddings = {}
    
    # Process each person directory
    for person_id in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_id)
        if not os.path.isdir(person_dir):
            continue
        
        fr_dir = os.path.join(person_dir, "frontal")
        pr_dir = os.path.join(person_dir, "profile")
        
        if not (os.path.isdir(fr_dir) and os.path.isdir(pr_dir)):
            print(f"=> Skipping {person_id}: missing 'frontal' or 'profile' folder")
            continue
        
        fr_imgs = get_image_paths(fr_dir)
        pr_imgs = get_image_paths(pr_dir)
        
        if not fr_imgs or not pr_imgs:
            print(f"=> Skipping {person_id}: no images in one of the subfolders")
            continue
        
        # Extract embeddings
        emb_fr = extract_embeddings(model, model_name, fr_imgs, device)
        emb_pr = extract_embeddings(model, model_name, pr_imgs, device)
        
        if emb_fr is None or emb_pr is None:
            print(f"=> Skipping {person_id}: failed to extract embeddings")
            continue
        
        # Normalize embeddings
        emb_fr = l2_normalize(emb_fr)
        emb_pr = l2_normalize(emb_pr)
        
        # Store embeddings for cross-person comparisons
        person_embeddings[person_id] = {
            'frontal': emb_fr,
            'profile': emb_pr
        }
        
        # Compute metrics for same person
        n_ff, mean_ff, std_ff, min_ff, max_ff, ver_ff = compute_pairwise_metrics(
            emb_fr, emb_fr, threshold, within=True)
        n_pp, mean_pp, std_pp, min_pp, max_pp, ver_pp = compute_pairwise_metrics(
            emb_pr, emb_pr, threshold, within=True)
        n_fp, mean_fp, std_fp, min_fp, max_fp, ver_fp = compute_pairwise_metrics(
            emb_fr, emb_pr, threshold, within=False)
        
        # Save FF and PP similarities for ROC calculation
        if n_ff > 0:
            sims_ff = cosine_similarity(emb_fr, emb_fr)
            i, j = np.triu_indices(sims_ff.shape[0], k=1)
            all_same_sims.extend(sims_ff[i, j].flatten())
        
        if n_pp > 0:
            sims_pp = cosine_similarity(emb_pr, emb_pr)
            i, j = np.triu_indices(sims_pp.shape[0], k=1)
            all_same_sims.extend(sims_pp[i, j].flatten())
        
        # Save FP similarities
        if n_fp > 0:
            sims_fp = cosine_similarity(emb_fr, emb_pr).flatten()
            all_same_sims.extend(sims_fp)
        
        # Print results for this person
        print(f"{person_id:>4}  | "
              f"FF pairs={n_ff:3d}, u={mean_ff:.3f}, acc={ver_ff*100:5.1f}%  | "
              f"PP pairs={n_pp:3d}, u={mean_pp:.3f}, acc={ver_pp*100:5.1f}%  | "
              f"FP pairs={n_fp:3d}, u={mean_fp:.3f}, acc={ver_fp*100:5.1f}%")
        
        # Record results
        results.append({
            "person": person_id,
            "ff_pairs": n_ff, "ff_mean": mean_ff, "ff_std": std_ff, 
            "ff_min": min_ff, "ff_max": max_ff, "ff_acc": ver_ff,
            "pp_pairs": n_pp, "pp_mean": mean_pp, "pp_std": std_pp, 
            "pp_min": min_pp, "pp_max": max_pp, "pp_acc": ver_pp,
            "fp_pairs": n_fp, "fp_mean": mean_fp, "fp_std": std_fp, 
            "fp_min": min_fp, "fp_max": max_fp, "fp_acc": ver_fp,
        })
    
    # Compute cross-person (different identity) similarities
    person_ids = list(person_embeddings.keys())
    for i in range(len(person_ids)):
        for j in range(i+1, len(person_ids)):
            person1 = person_ids[i]
            person2 = person_ids[j]
            
            # Compare frontal-frontal across different people
            sims_ff_diff = cosine_similarity(
                person_embeddings[person1]['frontal'],
                person_embeddings[person2]['frontal']
            ).flatten()
            all_diff_sims.extend(sims_ff_diff)
            
            # Compare profile-profile across different people
            sims_pp_diff = cosine_similarity(
                person_embeddings[person1]['profile'],
                person_embeddings[person2]['profile']
            ).flatten()
            all_diff_sims.extend(sims_pp_diff)
            
            # Compare frontal-profile across different people
            sims_fp_diff1 = cosine_similarity(
                person_embeddings[person1]['frontal'],
                person_embeddings[person2]['profile']
            ).flatten()
            all_diff_sims.extend(sims_fp_diff1)
            
            # Compare profile-frontal across different people
            sims_fp_diff2 = cosine_similarity(
                person_embeddings[person1]['profile'],
                person_embeddings[person2]['frontal']
            ).flatten()
            all_diff_sims.extend(sims_fp_diff2)
    
    return results, np.array(all_same_sims), np.array(all_diff_sims)

def main():
    parser = argparse.ArgumentParser(description='Benchmark all face recognition models')
    parser.add_argument('--data_dir', type=str, default='cfp-dataset/Data/Images', 
                        help='Path to CFP dataset images')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', 
                        help='Directory to save results')
    parser.add_argument('--elasticface_path', type=str, default='pretrain-model/295672backbone.pth',
                        help='Path to ElasticFace model')
    parser.add_argument('--adaface_path', type=str, default='pretrain-model/adaface_ir101_webface12m.ckpt',
                        help='Path to AdaFace model')
    parser.add_argument('--facenet512_path', type=str, default='pretrain-model/facenet512_model.h5',
                        help='Path to FaceNet512 model')
    parser.add_argument('--buffalo_l_path', type=str, default='pretrain-model/buffalo_l',
                        help='Path to Buffalo_L model')
    parser.add_argument('--use_demo', action='store_true',
                        help='Use demo models for ElasticFace and AdaFace')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated list of models to test (elasticface,adaface,facenet512,buffalo_l) or "all"')
    parser.add_argument('--thresholds', type=str, default='0.8,0.8,0.8,0.8',
                        help='Comma-separated list of thresholds for each model (elasticface,adaface,facenet512,buffalo_l)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device to GPU if available
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    if use_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Define model configurations
    model_configs = {
        'elasticface': {
            'path': 'pretrain-model/elasticface_demo.ckpt' if args.use_demo else args.elasticface_path,
            'threshold': 0.3,
            'load_func': load_elasticface
        },
        'adaface': {
            'path': 'pretrain-model/adaface_demo.ckpt' if args.use_demo else args.adaface_path,
            'threshold': 0.4,
            'load_func': load_adaface
        },
        'facenet512': {
            'path': args.facenet512_path,
            'threshold': 0.4,
            'load_func': load_facenet512
        },
        'buffalo_l': {
            'path': args.buffalo_l_path.replace('/', os.path.sep).replace('\\', os.path.sep),
            'threshold': 0.5,
            'load_func': load_buffalo_l
        }
    }
    
    # Parse threshold values
    thresholds = [float(t) for t in args.thresholds.split(',')]
    if len(thresholds) >= 4:
        model_configs['elasticface']['threshold'] = thresholds[0]
        model_configs['adaface']['threshold'] = thresholds[1]
        model_configs['facenet512']['threshold'] = thresholds[2]
        model_configs['buffalo_l']['threshold'] = thresholds[3]
    
    # Determine which models to test
    if args.models.lower() == 'all':
        models_to_test = list(model_configs.keys())
    else:
        models_to_test = [m.strip() for m in args.models.split(',')]
        # Filter out invalid models
        models_to_test = [m for m in models_to_test if m in model_configs]
    
    if not models_to_test:
        print("Error: No valid models specified.")
        return
    
    # Initialize dictionaries to store results
    all_results = {}
    verification_metrics = {}
    roc_data = {}
    
    # Benchmark each model
    for model_name in models_to_test:
        config = model_configs[model_name]
        
        print(f"\n{'='*20} Testing {model_name.upper()} {'='*20}")
        
        # Check if model file exists
        if not os.path.exists(config['path']):
            if model_name in ['facenet512', 'buffalo_l']:
                print(f"Note: {model_name} will download model weights automatically.")
            else:
                print(f"Error: Model file not found at {config['path']}")
                continue
        
        # Load model
        try:
            print(f"Loading {model_name} model...")
            start_time = time.time()
            
            # Pass device parameter to PyTorch models
            if model_name in ['elasticface', 'adaface']:
                model = config['load_func'](config['path'], device=device)
            else:
                model = config['load_func'](config['path'])
            
            # Move model to appropriate device if it's a PyTorch model
            if model_name in ['elasticface', 'adaface'] and isinstance(model, torch.nn.Module):
                model.eval()
            
            load_time = time.time() - start_time
            print(f"{model_name} model loaded in {load_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            continue
        
        # Run benchmark
        print(f"Benchmarking {model_name} on {args.data_dir}...")
        start_time = time.time()
        
        results, same_sims, diff_sims = benchmark_model(
            model, model_name, args.data_dir, config['threshold'], device
        )
        
        processing_time = time.time() - start_time
        print(f"{model_name} benchmark completed in {processing_time:.2f} seconds")
        
        # Calculate verification metrics at different thresholds
        metrics_df, roc_auc, eer, eer_threshold, fpr, tpr, precision, recall = calculate_verification_metrics(
            same_sims, diff_sims
        )
        
        # Store results
        all_results[model_name] = pd.DataFrame(results)
        verification_metrics[model_name] = {
            'metrics': metrics_df,
            'same_sims': same_sims,
            'diff_sims': diff_sims,
            'processing_time': processing_time
        }
        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'precision': precision,
            'recall': recall
        }
        
        # Print summary
        print(f"\n=== {model_name.upper()} Summary ===")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
        print(f"FF accuracy: {all_results[model_name]['ff_acc'].mean()*100:.2f}%")
        print(f"PP accuracy: {all_results[model_name]['pp_acc'].mean()*100:.2f}%")
        print(f"FP accuracy: {all_results[model_name]['fp_acc'].mean()*100:.2f}%")
        
        # Save results to CSV
        all_results[model_name].to_csv(os.path.join(args.output_dir, f"{model_name}_person_results.csv"), index=False)
        verification_metrics[model_name]['metrics'].to_csv(os.path.join(args.output_dir, f"{model_name}_verification_metrics.csv"), index=False)
    
    # Create a summary DataFrame
    summary_data = []
    
    for model_name in models_to_test:
        if model_name in all_results:
            df = all_results[model_name]
            summary_data.append({
                'Model': model_name,
                'FF_Accuracy': df['ff_acc'].mean() * 100,
                'PP_Accuracy': df['pp_acc'].mean() * 100,
                'FP_Accuracy': df['fp_acc'].mean() * 100,
                'FF_Similarity': df['ff_mean'].mean(),
                'PP_Similarity': df['pp_mean'].mean(),
                'FP_Similarity': df['fp_mean'].mean(),
                'ROC_AUC': roc_data[model_name]['auc'],
                'EER': roc_data[model_name]['eer'],
                'Optimal_Threshold': roc_data[model_name]['eer_threshold'],
                'Processing_Time': verification_metrics[model_name]['processing_time']
            })
    
    # Create and save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(args.output_dir, 'model_comparison_summary.csv'), index=False)
        print(f"\nSummary saved to {os.path.join(args.output_dir, 'model_comparison_summary.csv')}")
        print("\nTo visualize results, run:")
        print(f"python visualize_benchmark_results.py --benchmark_dir {args.output_dir}")

if __name__ == '__main__':
    main() 